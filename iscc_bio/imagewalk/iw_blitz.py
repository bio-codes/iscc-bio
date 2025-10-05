# -*- coding: utf-8 -*-
"""Blitz/OMERO implementation of IMAGEWALK plane traversal.

This module provides deterministic plane traversal for multi-dimensional bioimage data
stored in OMERO servers, conforming to the IMAGEWALK specification.
"""

from typing import Generator
from iscc_bio.imagewalk.models import Plane
from loguru import logger
import numpy as np


def iter_planes_blitz_fileset(conn, fileset):
    # type: (object, object) -> Generator[Plane, None, None]
    """Iterate over 2D planes in all images of an OMERO fileset following IMAGEWALK traversal order.

    Processes all images in a fileset (corresponding to a bioimage file) and yields planes
    in deterministic order. Each image in the fileset becomes a separate scene.

    Conforms to IMAGEWALK specification for deterministic bioimage traversal.

    :param conn: BlitzGateway connection to OMERO server
    :param fileset: OMERO Fileset object to process
    :return: Generator yielding Plane objects in Z→C→T order for all images
    """
    # Get all images in the fileset
    images = list(fileset.copyImages())

    fileset_id = fileset.getId()
    logger.debug(f"OMERO Fileset {fileset_id} - processing {len(images)} image(s)")

    # Process each image as a scene
    for scene_idx, image in enumerate(images):
        # Yield all planes from this image/scene
        yield from iter_planes_blitz_image(conn, image, scene_idx)


def iter_planes_blitz_image(conn, image, scene_idx=0):
    # type: (object, object, int) -> Generator[Plane, None, None]
    """Iterate over 2D planes in a single OMERO image following IMAGEWALK Z→C→T traversal order.

    Processes a single OMERO image (corresponding to a scene) and yields planes in
    deterministic order:
    - Outermost loop: Z dimension (depth/focal plane)
    - Middle loop: C dimension (channel)
    - Innermost loop: T dimension (time)

    Conforms to IMAGEWALK specification for deterministic bioimage traversal.

    :param conn: BlitzGateway connection to OMERO server
    :param image: OMERO Image object to process
    :param scene_idx: Scene index for the Plane objects (default: 0)
    :return: Generator yielding Plane objects in Z→C→T order
    """
    logger.debug(
        f"OMERO - processing scene {scene_idx}: {image.getName()} (ID: {image.getId()})"
    )

    # Get pixels object and ID
    pixels = image.getPrimaryPixels()
    pixels_id = pixels.getId()

    # Get dimensions
    size_t = image.getSizeT()
    size_c = image.getSizeC()
    size_z = image.getSizeZ()
    size_y = image.getSizeY()
    size_x = image.getSizeX()

    logger.debug(
        f"OMERO - scene {scene_idx}: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
    )

    # Get pixel data type mapping
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

    # Create RawPixelsStore service for proper plane access
    rps = conn.c.sf.createRawPixelsStore()

    try:
        # Set the pixels ID to access the pixel data
        rps.setPixelsId(pixels_id, True)  # True = bypass cache

        # Traverse planes in Z→C→T order (IMAGEWALK specification)
        for z in range(size_z):
            for c in range(size_c):
                for t in range(size_t):
                    # Get 2D plane using RawPixelsStore
                    plane_bytes = rps.getPlane(z, c, t)

                    # Convert to numpy array
                    plane_array = np.frombuffer(plane_bytes, dtype=np_dtype)
                    xy_array = plane_array.reshape(size_y, size_x)

                    # Yield Plane object
                    yield Plane(
                        xy_array=xy_array,
                        scene_idx=scene_idx,
                        z_depth=z,
                        c_channel=c,
                        t_time=t,
                    )
    finally:
        # Always close the RawPixelsStore
        rps.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m iscc_bio.imagewalk.iw_blitz <server_url> <image_id|fileset_id> [--fileset]"
        )
        sys.exit(1)

    from omero.gateway import BlitzGateway

    server_url = sys.argv[1]
    object_id = int(sys.argv[2])
    use_fileset = "--fileset" in sys.argv

    # Configure logger to show debug messages
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # Connect to OMERO (using hardcoded credentials for testing)
    logger.info(f"Connecting to OMERO server: {server_url}")
    conn = BlitzGateway("root", "omero", host=server_url, port=4064)

    if not conn.connect():
        logger.error(f"Failed to connect to OMERO server: {server_url}")
        sys.exit(1)

    try:
        plane_count = 0

        if use_fileset:
            # Process entire fileset
            logger.info(f"Processing OMERO fileset: {object_id}")
            fileset = conn.getObject("Fileset", object_id)
            if not fileset:
                logger.error(f"Fileset {object_id} not found")
                sys.exit(1)

            for plane in iter_planes_blitz_fileset(conn, fileset):
                plane_count += 1
                logger.info(
                    f"Plane {plane_count}: scene={plane.scene_idx}, z={plane.z_depth}, "
                    f"c={plane.c_channel}, t={plane.t_time}, shape={plane.xy_array.shape}, "
                    f"dtype={plane.xy_array.dtype}"
                )
        else:
            # Process single image
            logger.info(f"Processing OMERO image: {object_id}")
            image = conn.getObject("Image", object_id)
            if not image:
                logger.error(f"Image {object_id} not found")
                sys.exit(1)

            for plane in iter_planes_blitz_image(conn, image):
                plane_count += 1
                logger.info(
                    f"Plane {plane_count}: scene={plane.scene_idx}, z={plane.z_depth}, "
                    f"c={plane.c_channel}, t={plane.t_time}, shape={plane.xy_array.shape}, "
                    f"dtype={plane.xy_array.dtype}"
                )

        logger.info(f"Total planes processed: {plane_count}")

    finally:
        conn.close()
