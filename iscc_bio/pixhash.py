"""Normalized pixel hash generation for bioimaging data.

This module provides compatible implementations for generating reproducible ISCC-SUM hashes
over normalized pixel data from various bioimage sources (local files, OMERO server, OME-Zarr).
All implementations produce identical hashes for the same image data.
"""

import iscc_sum
import struct
from pathlib import Path
from typing import List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _plane_to_canonical_bytes(plane: np.ndarray) -> bytes:
    """Convert a 2D plane to canonical byte representation.

    Uses big-endian byte order for compatibility with OMERO.

    Args:
        plane: 2D NumPy array representing a single plane

    Returns:
        Bytes in big-endian format
    """
    if plane.ndim != 2:
        raise ValueError(f"Expected 2D plane, got {plane.ndim}D")

    # Get the struct format character
    dtype_map = {
        np.dtype("uint8"): "B",
        np.dtype("uint16"): "H",
        np.dtype("uint32"): "I",
        np.dtype("int8"): "b",
        np.dtype("int16"): "h",
        np.dtype("int32"): "i",
        np.dtype("float32"): "f",
        np.dtype("float64"): "d",
    }

    format_char = dtype_map.get(plane.dtype)
    if not format_char:
        raise ValueError(f"Unsupported dtype: {plane.dtype}")

    # Flatten plane in C-order (row-major: Y then X)
    flat = plane.flatten(order="C")

    # Pack to bytes in big-endian format
    format_str = f">{len(flat)}{format_char}"
    canonical_bytes = struct.pack(format_str, *flat)

    return canonical_bytes


def pixhash_bioio(image_path: str) -> List[str]:
    """Generate ISCC-SUM hashes for each scene in a bioimage file using BioIO.

    Args:
        image_path: Path to the bioimage file

    Returns:
        List of ISCC-SUM hash strings (one per scene)
    """
    from bioio import BioImage

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    hashes = []
    img = BioImage(image_path)

    # Process each scene
    num_scenes = len(img.scenes)
    logger.info(f"Processing {num_scenes} scene(s) from {image_path.name}")

    for scene_idx in range(num_scenes):
        if num_scenes > 1:
            img.set_scene(scene_idx)
            logger.info(f"Processing scene {scene_idx}: {img.scenes[scene_idx]}")

        # Get dimensions
        dims = img.dims
        shape = dims.shape
        dim_order = dims.order

        # Find dimension indices
        t_idx = dim_order.index("T") if "T" in dim_order else None
        c_idx = dim_order.index("C") if "C" in dim_order else None
        z_idx = dim_order.index("Z") if "Z" in dim_order else None
        y_idx = dim_order.index("Y")
        x_idx = dim_order.index("X")

        # Get dimension sizes
        size_t = shape[t_idx] if t_idx is not None else 1
        size_c = shape[c_idx] if c_idx is not None else 1
        size_z = shape[z_idx] if z_idx is not None else 1
        size_y = shape[y_idx]
        size_x = shape[x_idx]

        logger.debug(
            f"Scene {scene_idx} dimensions: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
        )

        # Initialize hasher for this scene
        hasher = iscc_sum.IsccSumProcessor()

        # Process planes in Z→C→T order (OMERO XYZCT storage order)
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

                    # Get 2D plane
                    plane = img.get_image_data("YX", **kwargs)

                    # Convert to canonical bytes and update hash
                    canonical_bytes = _plane_to_canonical_bytes(plane)
                    hasher.update(canonical_bytes)

        # Get final hash for this scene
        scene_hash = hasher.result(wide=True, add_units=False).iscc
        hashes.append(scene_hash)
        logger.info(f"Scene {scene_idx}: {scene_hash}")

    return hashes


def pixhash_omero(server_url: str, image_id: int) -> List[str]:
    """Generate ISCC-SUM hashes for all images in an OMERO OriginalFile.

    Args:
        server_url: OMERO server URL (e.g., "omero.server.com")
        image_id: OMERO image ID

    Returns:
        List of ISCC-SUM hash strings (one per image in the OriginalFile)
    """
    from omero.gateway import BlitzGateway

    # Connect to OMERO (using hardcoded credentials for now)
    logger.info(f"Connecting to OMERO server: {server_url}")
    conn = BlitzGateway("root", "omero", host=server_url, port=4064)

    if not conn.connect():
        raise ConnectionError(f"Failed to connect to OMERO server: {server_url}")

    try:
        # Get the image object
        image = conn.getObject("Image", image_id)
        if not image:
            raise ValueError(f"Image {image_id} not found on server")

        logger.info(f"Found image: {image.getName()} (ID: {image_id})")

        # Get the fileset to find all images from the same OriginalFile
        fileset = image.getFileset()
        hashes = []

        if fileset:
            # Get all images in the fileset
            images = list(fileset.copyImages())
            logger.info(f"Found {len(images)} image(s) in fileset")
        else:
            # Single image
            images = [image]
            logger.info("Processing single image (no fileset)")

        # Process each image
        for img in images:
            logger.info(f"Processing image: {img.getName()} (ID: {img.getId()})")

            # Get pixels object
            pixels = img.getPrimaryPixels()
            pixels_id = pixels.getId()

            # Get dimensions
            size_t = img.getSizeT()
            size_c = img.getSizeC()
            size_z = img.getSizeZ()
            size_y = img.getSizeY()
            size_x = img.getSizeX()

            logger.debug(
                f"Image dimensions: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
            )

            # Initialize hasher for this image
            hasher = iscc_sum.IsccSumProcessor()

            # Process planes in Z→C→T order (OMERO XYZCT storage order)
            for z in range(size_z):
                for c in range(size_c):
                    for t in range(size_t):
                        # Get 2D plane using OMERO's getPrimaryPixels
                        plane = pixels.getPlane(z, c, t)

                        # Convert to numpy array with appropriate dtype
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
                        plane_array = np.frombuffer(plane, dtype=np_dtype)
                        plane_array = plane_array.reshape(size_y, size_x)

                        # Convert to canonical bytes and update hash
                        canonical_bytes = _plane_to_canonical_bytes(plane_array)
                        hasher.update(canonical_bytes)

            # Get final hash for this image
            image_hash = hasher.result(wide=True, add_units=False).iscc
            hashes.append(image_hash)
            logger.info(f"Image {img.getId()}: {image_hash}")

        return hashes

    finally:
        conn.close()


def pixhash_zarr(zarr_path: str) -> List[str]:
    """Generate ISCC-SUM hashes for each series in an OME-Zarr file.

    Args:
        zarr_path: Path to the OME-Zarr file/directory

    Returns:
        List of ISCC-SUM strings (one per series)
    """
    import zarr

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"OME-Zarr file not found: {zarr_path}")

    hashes = []

    # Open the zarr group
    root_group = zarr.open(str(zarr_path), mode="r")

    # Check for series directories (0, 1, 2, etc.)
    series_dirs = []
    for key in root_group.keys():
        if key.isdigit():
            series_dirs.append(key)

    if not series_dirs:
        logger.warning("No series found in OME-Zarr file")
        return hashes

    logger.info(f"Found {len(series_dirs)} series in OME-Zarr")

    # Process each series
    for series_idx, series_key in enumerate(sorted(series_dirs, key=int)):
        logger.info(f"Processing series {series_key}")

        series_group = root_group[series_key]

        # Find the highest resolution level (usually '0')
        resolution_levels = []
        for key in series_group.keys():
            if key.isdigit():
                resolution_levels.append(key)

        if not resolution_levels:
            logger.warning(f"No resolution levels found in series {series_key}")
            continue

        # Use the highest resolution (smallest number, typically '0')
        highest_res = min(resolution_levels, key=int)
        data = series_group[highest_res]

        # Get metadata to determine dimension order
        attrs = series_group.attrs.asdict()

        # Try to get dimension order from multiscales metadata
        dim_order = None
        if "multiscales" in attrs and attrs["multiscales"]:
            multiscale = attrs["multiscales"][0]
            if "axes" in multiscale:
                axes = multiscale["axes"]
                # Axes might have 'name' or 'type' fields
                dim_order = ""
                for ax in axes:
                    if "name" in ax:
                        dim_order += ax["name"].upper()
                    elif "type" in ax:
                        # Map type to dimension name
                        type_map = {"time": "T", "channel": "C", "space": ""}
                        ax_type = ax["type"]
                        if ax_type == "space":
                            # Need to determine which spatial dimension
                            if len(dim_order) >= len(axes) - 2:
                                dim_order += (
                                    "Y" if len(dim_order) == len(axes) - 2 else "X"
                                )
                            else:
                                # Default spatial order
                                dim_order += (
                                    "Z"
                                    if len(
                                        [a for a in axes if a.get("type") == "space"]
                                    )
                                    > 2
                                    else ""
                                )
                        else:
                            dim_order += type_map.get(ax_type, "")

        # Fall back to common patterns if metadata is unclear
        if not dim_order:
            shape_len = len(data.shape)
            if shape_len == 5:
                dim_order = "TCZYX"
            elif shape_len == 4:
                dim_order = "CZYX"  # or could be TZYX
            elif shape_len == 3:
                dim_order = "CYX"  # or could be ZYX
            elif shape_len == 2:
                dim_order = "YX"
            else:
                logger.warning(f"Unexpected shape dimensions: {shape_len}")
                continue

        logger.info(f"Series {series_key} dimension order: {dim_order}")
        logger.info(f"Series {series_key} shape: {data.shape}")

        # Map dimensions
        shape = data.shape
        t_idx = dim_order.index("T") if "T" in dim_order else None
        c_idx = dim_order.index("C") if "C" in dim_order else None
        z_idx = dim_order.index("Z") if "Z" in dim_order else None
        y_idx = dim_order.index("Y") if "Y" in dim_order else None
        x_idx = dim_order.index("X") if "X" in dim_order else None

        # Get dimension sizes
        size_t = shape[t_idx] if t_idx is not None else 1
        size_c = shape[c_idx] if c_idx is not None else 1
        size_z = shape[z_idx] if z_idx is not None else 1
        size_y = shape[y_idx] if y_idx is not None else shape[-2]
        size_x = shape[x_idx] if x_idx is not None else shape[-1]

        logger.debug(
            f"Series {series_key} dimensions: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
        )

        # Initialize hasher for this series
        hasher = iscc_sum.IsccSumProcessor()

        # Process planes in Z→C→T order (matching OMERO)
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

                    # Get 2D plane
                    plane = data[tuple(indices)]

                    # Ensure it's 2D (squeeze out singleton dimensions)
                    plane = np.squeeze(plane)
                    if plane.ndim != 2:
                        # If still not 2D, take the last 2 dimensions
                        while plane.ndim > 2:
                            plane = plane[0]

                    # Convert to canonical bytes and update hash
                    canonical_bytes = _plane_to_canonical_bytes(plane)
                    hasher.update(canonical_bytes)

        # Get final hash for this series
        series_hash = hasher.result(wide=True, add_units=False).iscc
        hashes.append(series_hash)
        logger.info(f"Series {series_key}: {series_hash}")

    return hashes
