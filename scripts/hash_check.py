"""Minimal script to check for OMERO stored SHA1 vs calculated SHA1 mismatches."""

import argparse
import logging
from omero.gateway import BlitzGateway

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_image_hash(conn, image_id):
    """Check if stored SHA1 matches calculated SHA1 for an image."""
    image = conn.getObject("Image", image_id)
    if not image:
        logger.warning(f"Image {image_id} not found")
        return None

    pixels = image.getPrimaryPixels()
    stored_sha1 = pixels.getSha1()
    pixels_id = pixels.getId()

    rps = conn.c.sf.createRawPixelsStore()
    try:
        rps.setPixelsId(pixels_id, True)
        digest_bytes = rps.calculateMessageDigest()
        calculated_sha1 = digest_bytes.hex()
    finally:
        rps.close()

    match = stored_sha1 == calculated_sha1

    if match:
        logger.info(f"Image {image_id}: ✓ MATCH - {stored_sha1}")
    else:
        logger.error(f"Image {image_id}: ✗ MISMATCH")
        logger.error(f"  Stored:     {stored_sha1}")
        logger.error(f"  Calculated: {calculated_sha1}")

    return match


def main():
    parser = argparse.ArgumentParser(
        description="Check OMERO stored SHA1 vs calculated SHA1"
    )
    parser.add_argument("--host", default="omero.iscc.id", help="OMERO server host")
    parser.add_argument("--user", default="root", help="OMERO username")
    parser.add_argument("--password", default="omero", help="OMERO password")
    parser.add_argument("--image-id", type=int, help="Specific image ID to check")

    args = parser.parse_args()

    conn = BlitzGateway(args.user, args.password, host=args.host, port=4064)

    if not conn.connect():
        logger.error(f"Failed to connect to {args.host}")
        return 1

    try:
        if args.image_id:
            # Check single image
            result = check_image_hash(conn, args.image_id)
            return 0 if result else 1
        else:
            # Check all images
            logger.info("Checking all images on server...")
            matches = 0
            mismatches = 0

            for image in conn.getObjects("Image"):
                result = check_image_hash(conn, image.getId())
                if result is True:
                    matches += 1
                elif result is False:
                    mismatches += 1

            logger.info(f"\nSummary: {matches} matches, {mismatches} mismatches")
            return 0 if mismatches == 0 else 1

    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    sys.exit(main())
