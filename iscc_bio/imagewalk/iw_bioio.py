# -*- coding: utf-8 -*-
"""BioIO implementation of IMAGEWALK plane traversal.

This module provides deterministic plane traversal for multi-dimensional bioimage data
using the BioIO library, conforming to the IMAGEWALK specification.
"""

from typing import Generator
from pathlib import Path
from iscc_bio.imagewalk.models import Plane
from loguru import logger
import bioio


def iter_planes_bioio(image):
    # type: (bioio.ImageLike) -> Generator[Plane, None, None]
    """Iterate over 2D planes in a bioimage following IMAGEWALK Z→C→T traversal order.

    Uses lazy loading via Dask arrays to avoid loading the entire image into memory.
    Processes each scene independently and yields planes in deterministic order:
    - Outermost loop: Z dimension (depth/focal plane)
    - Middle loop: C dimension (channel)
    - Innermost loop: T dimension (time)

    Conforms to IMAGEWALK specification for deterministic bioimage traversal.

    :param image: Path to bioimage file, fsspec URI, or array-like object
    :return: Generator yielding Plane objects in Z→C→T order
    """

    # Open image with BioIO
    img = bioio.BioImage(image)

    # Log image information
    image_name = Path(image).name if isinstance(image, (str, Path)) else "array"
    logger.debug(f"{image_name} - using bioio implementation")
    logger.debug(f"{image_name} - using {img._plugin.entrypoint.name} reader")

    # Process each scene
    num_scenes = len(img.scenes)
    logger.debug(f"{image_name} - processing {num_scenes} scene(s)")

    for scene_idx in range(num_scenes):
        # Set current scene
        if num_scenes > 1:
            img.set_scene(scene_idx)
            logger.debug(
                f"{image_name} - processing scene {scene_idx}: {img.scenes[scene_idx]}"
            )

        # Get dimension information
        dims = img.dims
        shape = dims.shape
        dim_order = dims.order

        # Find dimension indices
        t_idx = dim_order.index("T") if "T" in dim_order else None
        c_idx = dim_order.index("C") if "C" in dim_order else None
        z_idx = dim_order.index("Z") if "Z" in dim_order else None
        y_idx = dim_order.index("Y")
        x_idx = dim_order.index("X")

        # Get dimension sizes (Y and X are required, others optional)
        size_t = shape[t_idx] if t_idx is not None else 1
        size_c = shape[c_idx] if c_idx is not None else 1
        size_z = shape[z_idx] if z_idx is not None else 1
        size_y = shape[y_idx]
        size_x = shape[x_idx]

        logger.debug(
            f"{image_name} - scene {scene_idx}: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
        )

        # Traverse planes in Z→C→T order (IMAGEWALK specification)
        for z in range(size_z):
            for c in range(size_c):
                for t in range(size_t):
                    # Build kwargs for dimension selection
                    kwargs = {}
                    if z_idx is not None:
                        kwargs["Z"] = z
                    if c_idx is not None:
                        kwargs["C"] = c
                    if t_idx is not None:
                        kwargs["T"] = t

                    # Get 2D plane using lazy loading (Dask array)
                    # This returns a Dask array without loading data into memory
                    lazy_plane = img.get_image_dask_data("YX", **kwargs)

                    # Compute only this specific plane to load it into memory
                    xy_array = lazy_plane.compute()

                    # Yield Plane object
                    yield Plane(
                        xy_array=xy_array,
                        scene_idx=scene_idx,
                        z_depth=z,
                        c_channel=c,
                        t_time=t,
                    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m iscc_bio.imagewalk.iw_bioio <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Configure logger to show debug messages
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info(f"Processing image: {image_path}")

    # Iterate through planes and log information
    plane_count = 0
    for plane in iter_planes_bioio(image_path):
        plane_count += 1
        logger.info(
            f"Plane {plane_count}: scene={plane.scene_idx}, z={plane.z_depth}, "
            f"c={plane.c_channel}, t={plane.t_time}, shape={plane.xy_array.shape}, "
            f"dtype={plane.xy_array.dtype}"
        )

    logger.info(f"Total planes processed: {plane_count}")
