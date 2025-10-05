# -*- coding: utf-8 -*-
"""OME-NGFF/Zarr implementation of IMAGEWALK plane traversal.

This module provides deterministic plane traversal for multi-dimensional bioimage data
in OME-NGFF/Zarr format using the ome-zarr-py library, conforming to the IMAGEWALK specification.
"""

from typing import Generator, Union
from pathlib import Path
from iscc_bio.imagewalk.models import Plane
from loguru import logger
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Multiscales
import numpy as np


def iter_planes_ngff(zarr_path):
    # type: (Union[str, Path]) -> Generator[Plane, None, None]
    """Iterate over 2D planes in an OME-NGFF/Zarr image following IMAGEWALK Z→C→T traversal order.

    Uses lazy loading via Dask arrays to avoid loading the entire image into memory.
    Processes each scene independently and yields planes in deterministic order:
    - Outermost loop: Z dimension (depth/focal plane)
    - Middle loop: C dimension (channel)
    - Innermost loop: T dimension (time)

    Conforms to IMAGEWALK specification for deterministic bioimage traversal.

    :param zarr_path: Path to OME-NGFF/Zarr file or directory
    :return: Generator yielding Plane objects in Z→C→T order
    """

    # Parse the zarr location
    zarr_location = parse_url(str(zarr_path))

    # Log image information
    zarr_name = Path(zarr_path).name if isinstance(zarr_path, (str, Path)) else "zarr"
    logger.debug(f"{zarr_name} - using ome-ngff implementation")

    # Check if this is a bioformats2raw layout (with numbered subdirectories)
    import zarr

    root_group = zarr.open_group(str(zarr_path), mode="r")

    # Find series/scene directories (numbered directories like 0, 1, 2, etc.)
    series_dirs = []
    for key in root_group.keys():
        if key.isdigit():
            series_dirs.append(key)

    if series_dirs:
        # Process bioformats2raw layout
        logger.debug(
            f"{zarr_name} - detected bioformats2raw layout with {len(series_dirs)} series"
        )

        # Process each series in numerical order
        for scene_idx, series_key in enumerate(sorted(series_dirs, key=int)):
            series_path = Path(zarr_path) / series_key
            series_location = parse_url(str(series_path))

            # Create reader for this series
            reader = Reader(series_location)
            nodes = list(reader())

            if not nodes:
                logger.warning(f"{zarr_name} - no nodes found in series {series_key}")
                continue

            # Process the first node (should contain the multiscale data)
            node = nodes[0]

            # Check if this node contains multiscale image data
            multiscales_spec = None
            for spec in node.specs:
                if isinstance(spec, Multiscales):
                    multiscales_spec = spec
                    break

            if not multiscales_spec:
                logger.debug(f"{zarr_name} - skipping non-image series {series_key}")
                continue

            logger.debug(
                f"{zarr_name} - processing scene {scene_idx}: series {series_key}"
            )

            # Get axes metadata and data pyramid
            axes_metadata = node.metadata.get("axes", [])
            data_pyramid = node.data

            if not data_pyramid:
                logger.warning(f"{zarr_name} - no data found in series {series_key}")
                continue

            # Use highest resolution (first level)
            data = data_pyramid[0]

            # Build dimension index mapping
            dimension_indices = {}
            for i, axis in enumerate(axes_metadata):
                name = axis.get("name")
                if name:
                    dimension_indices[name.upper()] = i

            # Get dimension indices
            t_idx = dimension_indices.get("T")
            c_idx = dimension_indices.get("C")
            z_idx = dimension_indices.get("Z")
            y_idx = dimension_indices.get("Y")
            x_idx = dimension_indices.get("X")

            # Get dimension sizes from data shape
            shape = data.shape
            size_t = shape[t_idx] if t_idx is not None else 1
            size_c = shape[c_idx] if c_idx is not None else 1
            size_z = shape[z_idx] if z_idx is not None else 1
            size_y = (
                shape[y_idx] if y_idx is not None else shape[-2]
            )  # Fallback to second-to-last
            size_x = (
                shape[x_idx] if x_idx is not None else shape[-1]
            )  # Fallback to last

            logger.debug(
                f"{zarr_name} - scene {scene_idx}: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
            )

            # Traverse planes in Z→C→T order (IMAGEWALK specification)
            for z in range(size_z):
                for c in range(size_c):
                    for t in range(size_t):
                        # Build indices for data access
                        indices = [slice(None)] * len(shape)
                        if t_idx is not None:
                            indices[t_idx] = t
                        if c_idx is not None:
                            indices[c_idx] = c
                        if z_idx is not None:
                            indices[z_idx] = z

                        # Get 2D plane using lazy loading (Dask array)
                        # This returns a Dask array without loading data into memory
                        lazy_plane = data[tuple(indices)]

                        # Compute only this specific plane to load it into memory
                        xy_array = lazy_plane.compute()

                        # Ensure it's 2D (squeeze out singleton dimensions)
                        xy_array = np.squeeze(xy_array)
                        if xy_array.ndim != 2:
                            # If still not 2D, ensure we have Y and X dimensions
                            while xy_array.ndim > 2:
                                xy_array = xy_array[0]
                            if xy_array.ndim < 2:
                                logger.error(
                                    f"Could not extract 2D plane at z={z}, c={c}, t={t}"
                                )
                                continue

                        # Yield Plane object
                        yield Plane(
                            xy_array=xy_array,
                            scene_idx=scene_idx,
                            z_depth=z,
                            c_channel=c,
                            t_time=t,
                        )
    else:
        # Process standard OME-NGFF layout
        reader = Reader(zarr_location)
        nodes = list(reader())

        scene_idx = 0
        for node in nodes:
            # Check if this node contains multiscale image data
            multiscales_spec = None
            for spec in node.specs:
                if isinstance(spec, Multiscales):
                    multiscales_spec = spec
                    break

            if not multiscales_spec:
                logger.debug(
                    f"{zarr_name} - skipping non-image node at {node.zarr.path}"
                )
                continue

            logger.debug(
                f"{zarr_name} - processing scene {scene_idx}: {node.zarr.path}"
            )

            # Get axes metadata and data pyramid
            axes_metadata = node.metadata.get("axes", [])
            data_pyramid = node.data

            if not data_pyramid:
                logger.warning(f"{zarr_name} - no data found in scene {scene_idx}")
                continue

            # Use highest resolution (first level)
            data = data_pyramid[0]

            # Build dimension index mapping
            dimension_indices = {}
            for i, axis in enumerate(axes_metadata):
                name = axis.get("name")
                if name:
                    dimension_indices[name.upper()] = i

            # Get dimension indices
            t_idx = dimension_indices.get("T")
            c_idx = dimension_indices.get("C")
            z_idx = dimension_indices.get("Z")
            y_idx = dimension_indices.get("Y")
            x_idx = dimension_indices.get("X")

            # Get dimension sizes from data shape
            shape = data.shape
            size_t = shape[t_idx] if t_idx is not None else 1
            size_c = shape[c_idx] if c_idx is not None else 1
            size_z = shape[z_idx] if z_idx is not None else 1
            size_y = shape[y_idx] if y_idx is not None else shape[-2]
            size_x = shape[x_idx] if x_idx is not None else shape[-1]

            logger.debug(
                f"{zarr_name} - scene {scene_idx}: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
            )

            # Traverse planes in Z→C→T order (IMAGEWALK specification)
            for z in range(size_z):
                for c in range(size_c):
                    for t in range(size_t):
                        # Build indices for data access
                        indices = [slice(None)] * len(shape)
                        if t_idx is not None:
                            indices[t_idx] = t
                        if c_idx is not None:
                            indices[c_idx] = c
                        if z_idx is not None:
                            indices[z_idx] = z

                        # Get 2D plane using lazy loading (Dask array)
                        lazy_plane = data[tuple(indices)]

                        # Compute only this specific plane to load it into memory
                        xy_array = lazy_plane.compute()

                        # Ensure it's 2D (squeeze out singleton dimensions)
                        xy_array = np.squeeze(xy_array)
                        if xy_array.ndim != 2:
                            while xy_array.ndim > 2:
                                xy_array = xy_array[0]
                            if xy_array.ndim < 2:
                                logger.error(
                                    f"Could not extract 2D plane at z={z}, c={c}, t={t}"
                                )
                                continue

                        # Yield Plane object
                        yield Plane(
                            xy_array=xy_array,
                            scene_idx=scene_idx,
                            z_depth=z,
                            c_channel=c,
                            t_time=t,
                        )

            scene_idx += 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m iscc_bio.imagewalk.iw_ngff <zarr_path>")
        sys.exit(1)

    zarr_path = sys.argv[1]

    # Configure logger to show debug messages
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info(f"Processing OME-NGFF/Zarr: {zarr_path}")

    # Iterate through planes and log information
    plane_count = 0
    for plane in iter_planes_ngff(zarr_path):
        plane_count += 1
        logger.info(
            f"Plane {plane_count}: scene={plane.scene_idx}, z={plane.z_depth}, "
            f"c={plane.c_channel}, t={plane.t_time}, shape={plane.xy_array.shape}, "
            f"dtype={plane.xy_array.dtype}"
        )

    logger.info(f"Total planes processed: {plane_count}")
