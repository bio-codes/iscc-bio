"""Bioimage scene extraction."""

import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from bioio import BioImage
import logging

logger = logging.getLogger(__name__)


def extract_scenes(
    image_path: Path,
    max_size: Tuple[int, int] = (512, 512),
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Extract thumbnails from all scenes in a bioimage file using lazy loading.

    Args:
        image_path: Path to the bioimage file
        max_size: Maximum thumbnail dimensions (width, height)
        output_dir: Optional output directory for thumbnails.
                   If None, saves in same directory as source file

    Returns:
        List of paths to the saved thumbnails
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = image_path.parent

    logger.info(f"Extracting scenes from {image_path.name}")

    # Use lazy loading with dask
    try:
        img = BioImage(image_path, reader=None)  # Let bioio choose best reader
    except Exception as e:
        # Provide helpful error messages for unsupported formats
        if ".ims" in str(image_path).lower():
            logger.error(
                "IMS format not supported. Install bioio-bioformats and Java: pip install bioio-bioformats"
            )
        elif ".oir" in str(image_path).lower():
            logger.error(
                "OIR format not supported. Install bioio-bioformats and Java: pip install bioio-bioformats"
            )
        raise e

    # Get number of scenes
    num_scenes = len(img.scenes)
    logger.info(f"Found {num_scenes} scenes in {image_path.name}")

    if num_scenes == 0:
        logger.warning(f"No scenes found in {image_path.name}")
        return []

    thumbnail_paths = []

    for scene_idx in range(num_scenes):
        scene_start = time.time()

        # Set current scene
        img.set_scene(scene_idx)

        # Use simple numeric index for scene naming
        scene_name = f"scene_{scene_idx:03d}"

        # Create output filename with scene index
        output_filename = f"{image_path.stem}.{scene_name}.thumb.png"
        output_path = output_dir / output_filename

        logger.info(f"Processing scene {scene_idx + 1}/{num_scenes}: {scene_name}")

        # Get image dimensions
        dims = img.dims
        shape = dims.shape

        logger.debug(f"Scene {scene_name} dimensions: {dict(zip(dims.order, shape))}")

        # Check for available resolution levels and use lowest for quick thumbnail
        resolution_levels = img.resolution_levels

        if len(resolution_levels) > 1:
            # Use the lowest resolution available for fastest loading
            img.set_resolution_level(resolution_levels[-1])
            logger.debug(f"Using resolution level: {resolution_levels[-1]}")

        # Get dimensions at current resolution
        current_shape = img.dask_data.shape
        dim_dict = dict(zip(img.dims.order, current_shape))

        # Build kwargs for get_image_dask_data
        kwargs = {}

        # Select middle Z plane if Z dimension exists
        if "Z" in dim_dict and dim_dict["Z"] > 1:
            kwargs["Z"] = dim_dict["Z"] // 2
        elif "Z" in dim_dict:
            kwargs["Z"] = 0

        # Select first timepoint if T exists
        if "T" in dim_dict:
            kwargs["T"] = 0

        # Calculate downsampling factor for very large images
        y_size = dim_dict.get("Y", 0)
        x_size = dim_dict.get("X", 0)

        # If image is very large, downsample by slicing
        if y_size > 2000 or x_size > 2000:
            # Since final perceptual hash uses 32x32, we can downsample aggressively
            # 512 pixels is sufficient for thumbnail that will be reduced to 32x32
            target_size = 512
            y_stride = max(1, y_size // target_size)
            x_stride = max(1, x_size // target_size)
            logger.debug(f"Downsampling with stride Y:{y_stride}, X:{x_stride}")

            # Get the lazy dask array
            if "C" in dim_dict:
                # Get first channel only for speed
                kwargs["C"] = 0
                lazy_data = img.get_image_dask_data("YX", **kwargs)
            else:
                lazy_data = img.get_image_dask_data("YX", **kwargs)

            # Apply strided slicing on the lazy array
            lazy_data = lazy_data[::y_stride, ::x_stride]

            # For color, try to get RGB from first 3 channels
            if "C" in dim_dict and dim_dict["C"] >= 3:
                # Get all channels at once and slice - more efficient than multiple calls
                kwargs.pop("C", None)  # Remove C from kwargs to get all channels
                lazy_data = img.get_image_dask_data("CYX", **kwargs)

                # Apply downsampling to all channels at once
                lazy_data = lazy_data[
                    :3, ::y_stride, ::x_stride
                ]  # Get first 3 channels and downsample

                # Compute and transpose to YXC format
                data = lazy_data.compute()
                data = np.moveaxis(data, 0, -1)  # CYX -> YXC
            else:
                # Compute the downsampled grayscale data
                data = lazy_data.compute()
        else:
            # For smaller images, use original approach
            if "C" in dim_dict:
                if dim_dict["C"] >= 3:
                    # Get first 3 channels for RGB
                    lazy_data = img.get_image_dask_data("CYX", **kwargs)
                    # Select first 3 channels
                    lazy_data = lazy_data[:3]
                else:
                    # Get all channels
                    lazy_data = img.get_image_dask_data("CYX", **kwargs)
            else:
                # Single channel
                lazy_data = img.get_image_dask_data("YX", **kwargs)

            # Compute only the selected data
            data = lazy_data.compute()

            # Handle channel dimension
            if data.ndim == 3:  # CYX format
                if data.shape[0] >= 3:
                    # Use first 3 channels as RGB
                    data = np.moveaxis(data[:3], 0, -1)  # CYX -> YXC
                else:
                    # Use first channel only
                    data = data[0]  # Take first channel

        # Normalize data to 0-255 range
        if data.dtype != np.uint8:
            data_min = np.min(data)
            data_max = np.max(data)

            if data_max > data_min:
                data = ((data - data_min) / (data_max - data_min) * 255).astype(
                    np.uint8
                )
            else:
                data = np.zeros_like(data, dtype=np.uint8)

        # Create PIL Image
        if data.ndim == 2:
            # Grayscale
            pil_image = Image.fromarray(data, mode="L")
        elif data.ndim == 3 and data.shape[-1] == 3:
            # RGB
            pil_image = Image.fromarray(data, mode="RGB")
        else:
            # Fallback to grayscale
            if data.ndim == 3:
                data = data[:, :, 0]
            pil_image = Image.fromarray(data, mode="L")

        # Resize maintaining aspect ratio
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Save thumbnail
        pil_image.save(output_path, "PNG")

        scene_elapsed = time.time() - scene_start
        logger.info(
            f"{scene_name} extracted in {scene_elapsed:.2f} seconds: {output_path}"
        )

        thumbnail_paths.append(output_path)

    total_elapsed = time.time() - start_time
    logger.info(f"All {num_scenes} scenes extracted in {total_elapsed:.2f} seconds")

    return thumbnail_paths
