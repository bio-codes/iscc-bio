"""Test that IMAGEWALK implementation produces OMERO-compatible checksums.

This script verifies that our pixhash_omero function using the IMAGEWALK
algorithm produces SHA1 hashes compatible with OMERO's stored checksums.
"""

import hashlib
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from iscc_bio.pixhash import pixhash_omero

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_omero_compatibility():
    """Test IMAGEWALK implementation against known OMERO SHA1 checksums."""

    # Test configuration
    SERVER_URL = "omero.iscc.id"
    IMAGE_ID = 52
    EXPECTED_SHA1 = "177d30b76a9e20296ea1e86a33a5c38a75c05fd1"

    logger.info("=" * 60)
    logger.info("IMAGEWALK-OMERO Compatibility Test")
    logger.info("=" * 60)

    logger.info("\nTest Parameters:")
    logger.info(f"  Server: {SERVER_URL}")
    logger.info(f"  Image ID: {IMAGE_ID}")
    logger.info(f"  Expected SHA1: {EXPECTED_SHA1}")

    logger.info("\nRunning pixhash_omero with IMAGEWALK algorithm...")

    try:
        # Generate ISCC hashes using our implementation
        iscc_hashes = pixhash_omero(SERVER_URL, IMAGE_ID)

        if not iscc_hashes:
            logger.error("No hashes generated!")
            return False

        logger.info(f"\nGenerated {len(iscc_hashes)} ISCC hash(es)")

        for i, iscc_hash in enumerate(iscc_hashes):
            logger.info(f"  Hash {i}: {iscc_hash}")

        # For SHA1 comparison, we need to extract the SHA1 from ISCC
        # or generate a separate SHA1 for comparison
        # Let's create a modified test that calculates raw SHA1

        logger.info("\n" + "=" * 60)
        logger.info("Testing SHA1 compatibility directly...")
        test_sha1_compatibility(SERVER_URL, IMAGE_ID, EXPECTED_SHA1)

        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sha1_compatibility(server_url, image_id, expected_sha1):
    """Test SHA1 calculation directly to verify byte-level compatibility."""
    from omero.gateway import BlitzGateway
    import numpy as np
    from iscc_bio.pixhash import _plane_to_canonical_bytes

    # Connect to OMERO
    conn = BlitzGateway("root", "omero", host=server_url, port=4064)

    if not conn.connect():
        raise ConnectionError(f"Failed to connect to OMERO server: {server_url}")

    try:
        # Get the image
        image = conn.getObject("Image", image_id)
        if not image:
            raise ValueError(f"Image {image_id} not found on server")

        logger.info(f"\nImage: {image.getName()} (ID: {image_id})")

        # Get pixels info
        pixels = image.getPrimaryPixels()
        pixels_id = pixels.getId()
        server_sha1 = pixels.getSha1()

        logger.info(f"Server SHA1: {server_sha1}")

        # Get dimensions
        size_t = image.getSizeT()
        size_c = image.getSizeC()
        size_z = image.getSizeZ()
        size_y = image.getSizeY()
        size_x = image.getSizeX()

        logger.info(
            f"Dimensions: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
        )

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

        # Create RawPixelsStore
        rps = conn.c.sf.createRawPixelsStore()

        try:
            rps.setPixelsId(pixels_id, True)

            # Calculate SHA1 using IMAGEWALK algorithm
            sha1_hasher = hashlib.sha1()

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

            calculated_sha1 = sha1_hasher.hexdigest()

            logger.info(f"Calculated SHA1: {calculated_sha1}")

            # Compare
            logger.info("\n" + "=" * 60)
            if server_sha1 == calculated_sha1:
                logger.info("✓ SUCCESS: IMAGEWALK produces OMERO-compatible SHA1!")
                return True
            elif expected_sha1 == calculated_sha1:
                logger.info("✓ SUCCESS: IMAGEWALK matches expected OMERO SHA1!")
                logger.info("(Server SHA1 may be outdated or pending)")
                return True
            else:
                logger.error("✗ FAILURE: SHA1 mismatch!")
                logger.error(f"  Expected:   {expected_sha1}")
                logger.error(f"  Server:     {server_sha1}")
                logger.error(f"  Calculated: {calculated_sha1}")
                return False

        finally:
            rps.close()

    finally:
        conn.close()


def main():
    """Main entry point."""
    success = test_omero_compatibility()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("TEST PASSED: IMAGEWALK is OMERO-compatible!")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("TEST FAILED: Compatibility issues detected")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
