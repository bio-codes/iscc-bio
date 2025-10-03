"""CLI script to verify IMAGEWALK implementation produces OMERO-compatible checksums.

This script calculates SHA1 checksums using the IMAGEWALK algorithm and compares
them with OMERO server-side checksums for compatibility verification.

Usage:
    python scripts/imagewalk_check.py <omero-server-url> <image-id> [--user <username>] [--password <password>]
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
from omero.gateway import BlitzGateway

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from iscc_bio.pixhash import _plane_to_canonical_bytes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_imagewalk_sha1(server_url, image_id, username, password):
    """Calculate SHA1 using IMAGEWALK algorithm and compare with OMERO server hash."""

    # Connect to OMERO
    conn = BlitzGateway(username, password, host=server_url, port=4064)

    if not conn.connect():
        raise ConnectionError(f"Failed to connect to OMERO server: {server_url}")

    try:
        # Get the image
        image = conn.getObject("Image", image_id)
        if not image:
            raise ValueError(f"Image {image_id} not found on server")

        logger.info("=" * 60)
        logger.info("IMAGEWALK-OMERO Compatibility Check")
        logger.info("=" * 60)

        # Get pixels info
        pixels = image.getPrimaryPixels()
        pixels_id = pixels.getId()
        server_sha1 = pixels.getSha1()

        # Get dimensions
        size_t = image.getSizeT()
        size_c = image.getSizeC()
        size_z = image.getSizeZ()
        size_y = image.getSizeY()
        size_x = image.getSizeX()

        # Get pixel data type
        dtype_str = str(pixels.getPixelsType().getValue())
        dtype_map = {
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "float": np.float32,
            "double": np.float64,
        }
        np_dtype = dtype_map.get(dtype_str, np.uint8)

        logger.info(f"\nServer: {server_url}")
        logger.info(f"Image: {image.getName()} (ID: {image_id})")
        logger.info(f"Server SHA1: {server_sha1}")

        # Create RawPixelsStore
        rps = conn.c.sf.createRawPixelsStore()

        try:
            rps.setPixelsId(pixels_id, True)

            # Calculate SHA1 using IMAGEWALK algorithm
            sha1_hasher = hashlib.sha1()

            logger.info("\nProcessing planes in IMAGEWALK order (Z→C→T):")
            logger.info(
                f"Scene 0: {image.getName()} | {size_x}×{size_y}, Z={size_z}, C={size_c}, T={size_t}, type={dtype_str}"
            )

            plane_num = 1
            # Process planes in Z→C→T order (IMAGEWALK specification)
            for z in range(size_z):
                for c in range(size_c):
                    for t in range(size_t):
                        # Get plane data
                        plane_bytes = rps.getPlane(z, c, t)

                        # Convert to numpy array
                        plane_array = np.frombuffer(plane_bytes, dtype=np_dtype)
                        plane_array = plane_array.reshape(size_y, size_x)

                        # Convert to canonical bytes (big-endian)
                        canonical_bytes = _plane_to_canonical_bytes(plane_array)

                        # Update hash
                        sha1_hasher.update(canonical_bytes)

                        # Log plane processing
                        logger.info(f"  Plane {plane_num}:  z={z}, c={c}, t={t}")
                        plane_num += 1

            calculated_sha1 = sha1_hasher.hexdigest()

            logger.info(f"\nCalculated SHA1: {calculated_sha1}")

            # Also get the server's calculateMessageDigest if available
            # NOTE: calculateMessageDigest() executes SERVER-SIDE on the OMERO server.
            # The RawPixelsStore (rps) is a proxy (RawPixelsStorePrx) that sends a
            # RawAccessRequest("checksum") to the server, which calculates the SHA-1
            # hash without transferring pixel data to the client. This is the current
            # live calculation vs the stored sha1 value which may be incorrect due to
            # a possible OMERO bug. (Verified: ome/openmicroscopy & ome/omero-py codebases)
            server_calculated_sha1 = None
            try:
                digest_bytes = rps.calculateMessageDigest()
                server_calculated_sha1 = digest_bytes.hex()
                logger.info(
                    f"Server calculateMessageDigest(): {server_calculated_sha1}"
                )
            except Exception as e:
                logger.debug(f"Could not get calculateMessageDigest: {e}")

            # Compare
            logger.info("\n" + "=" * 60)
            if server_sha1 == calculated_sha1:
                logger.info("✓ SUCCESS: IMAGEWALK matches stored SHA1!")
                logger.info("=" * 60)
                return True
            elif server_calculated_sha1 and server_calculated_sha1 == calculated_sha1:
                logger.warning(
                    "⚠ PARTIAL SUCCESS: IMAGEWALK matches calculateMessageDigest()"
                )
                logger.warning("  but NOT the stored SHA1!")
                logger.warning(f"  Stored SHA1:             {server_sha1}")
                logger.warning(f"  calculateMessageDigest: {server_calculated_sha1}")
                logger.warning(f"  IMAGEWALK:              {calculated_sha1}")
                logger.warning("  This appears to be an OMERO bug with stored SHA1!")
                logger.warning("=" * 60)
                return True  # Consider this a success since we match the correct calculation
            else:
                logger.error("✗ FAILURE: SHA1 mismatch!")
                logger.error(f"  Stored SHA1:    {server_sha1}")
                if server_calculated_sha1:
                    logger.error(f"  Server calc:    {server_calculated_sha1}")
                logger.error(f"  IMAGEWALK:      {calculated_sha1}")
                logger.error("=" * 60)
                return False

        finally:
            rps.close()

    finally:
        conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify IMAGEWALK implementation produces OMERO-compatible SHA1 checksums"
    )
    parser.add_argument("server_url", help="OMERO server URL")
    parser.add_argument("image_id", type=int, help="OMERO image ID")
    parser.add_argument("--user", default="root", help="OMERO username (default: root)")
    parser.add_argument(
        "--password", default="omero", help="OMERO password (default: omero)"
    )

    args = parser.parse_args()

    try:
        success = calculate_imagewalk_sha1(
            args.server_url, args.image_id, args.user, args.password
        )

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
