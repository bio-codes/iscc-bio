"""
Comprehensive Bioimage Analysis for ISCC Processing

This script analyzes bioimage files to understand their structure and properties
for ISCC Mixed-Code generation planning.
"""

import os
from pathlib import Path
import numpy as np
import psutil
from typing import Dict, Any, List, Tuple
from bioio import BioImage


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def extract_2d_slices_for_iscc(
    bio_img: BioImage, max_slices: int = 5
) -> List[Tuple[str, np.ndarray]]:
    """
    Extract representative 2D slices for ISCC processing.

    Returns a list of (description, 2D_array) tuples.
    """
    slices = []
    shape = bio_img.shape
    dims = str(bio_img.dims)

    print("  Extracting 2D slices for ISCC processing...")
    print(f"  Image shape: {shape}, dims: {dims}")

    try:
        # Strategy: Extract slices that represent different aspects of the bioimage
        # 1. If multi-channel: extract first few channels
        # 2. If Z-stack: extract slices from beginning, middle, end
        # 3. If time series: extract few time points

        # Get multi-channel slices if available
        if "C" in dims:
            cyx_data = bio_img.get_image_data("CYX")
            num_channels = cyx_data.shape[0]
            print(f"  Found {num_channels} channels")

            for c in range(min(max_slices, num_channels)):
                slice_2d = cyx_data[c, :, :]
                slices.append((f"Channel_{c}", slice_2d))
                print(
                    f"    Channel {c}: {slice_2d.shape}, range={slice_2d.min()}-{slice_2d.max()}"
                )

        # Get Z-stack slices if available
        elif "Z" in dims and bio_img.shape[dims.find("Z")] > 1:
            zyx_data = bio_img.get_image_data("ZYX")
            num_z = zyx_data.shape[0]
            print(f"  Found {num_z} Z-slices")

            # Extract beginning, middle, end
            z_indices = [0, num_z // 2, num_z - 1] if num_z > 2 else [0]
            z_indices = z_indices[:max_slices]

            for z in z_indices:
                slice_2d = zyx_data[z, :, :]
                slices.append((f"Z_slice_{z}", slice_2d))
                print(
                    f"    Z-slice {z}: {slice_2d.shape}, range={slice_2d.min()}-{slice_2d.max()}"
                )

        # Single 2D image
        else:
            yx_data = bio_img.get_image_data("YX")
            slices.append(("Single_2D", yx_data))
            print(
                f"    Single 2D: {yx_data.shape}, range={yx_data.min()}-{yx_data.max()}"
            )

    except Exception as e:
        print(f"  Error extracting slices: {e}")

    return slices


def analyze_for_iscc_processing(file_path: str) -> Dict[str, Any]:
    """
    Analyze bioimage file specifically for ISCC Mixed-Code generation.
    """
    print(f"\n{'=' * 70}")
    print(f"ISCC ANALYSIS: {Path(file_path).name}")
    print(f"File size: {format_size(os.path.getsize(file_path))}")
    print(f"{'=' * 70}")

    try:
        bio_img = BioImage(file_path)
        shape = bio_img.shape
        dims = str(bio_img.dims)
        dtype = bio_img.dtype

        # Basic information
        print(f"Shape: {shape}")
        print(f"Dimensions: {dims}")
        print(f"Data type: {dtype}")

        # Memory estimation
        total_pixels = np.prod(shape)
        bytes_per_pixel = dtype.itemsize
        total_bytes = total_pixels * bytes_per_pixel
        estimated_memory = format_size(total_bytes)
        print(f"Estimated memory: {estimated_memory}")

        # Physical pixel sizes
        physical_info = {}
        if bio_img.physical_pixel_sizes:
            pps = bio_img.physical_pixel_sizes
            if hasattr(pps, "Z") and pps.Z is not None:
                physical_info["Z"] = float(pps.Z)
            if hasattr(pps, "Y") and pps.Y is not None:
                physical_info["Y"] = float(pps.Y)
            if hasattr(pps, "X") and pps.X is not None:
                physical_info["X"] = float(pps.X)
            print(f"Physical pixel sizes: {physical_info}")

        # Channel information
        channel_info = {}
        if hasattr(bio_img, "channel_names") and bio_img.channel_names:
            channel_info["names"] = [str(name) for name in bio_img.channel_names]
            channel_info["count"] = len(channel_info["names"])
            print(f"Channels ({channel_info['count']}): {channel_info['names']}")

        # Extract representative 2D slices
        memory_before = get_memory_usage()
        slices = extract_2d_slices_for_iscc(bio_img, max_slices=3)
        memory_after = get_memory_usage()

        # Calculate potential ISCC processing requirements
        iscc_analysis = analyze_iscc_requirements(shape, dims, slices)

        return {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "shape": shape,
            "dimensions": dims,
            "dtype": str(dtype),
            "estimated_memory": estimated_memory,
            "physical_pixel_sizes": physical_info,
            "channel_info": channel_info,
            "extracted_slices": len(slices),
            "slice_extraction_memory": memory_after - memory_before,
            "iscc_analysis": iscc_analysis,
        }

    except Exception as e:
        print(f"Error analyzing file: {e}")
        return {"error": str(e), "file_path": file_path}


def analyze_iscc_requirements(
    shape: Tuple, dims: str, slices: List[Tuple[str, np.ndarray]]
) -> Dict[str, Any]:
    """
    Analyze requirements for ISCC Mixed-Code generation.
    """
    print("\n  ISCC Processing Analysis:")

    analysis = {
        "total_2d_images": 0,
        "processing_strategy": "",
        "memory_efficient": True,
        "estimated_iscc_codes": 0,
    }

    # Calculate number of 2D images for ISCC processing
    if "C" in dims and "Z" in dims:
        # Multi-channel Z-stack: C * Z images
        c_idx = dims.find("C")
        z_idx = dims.find("Z")
        num_channels = shape[c_idx] if c_idx < len(shape) else 1
        num_z = shape[z_idx] if z_idx < len(shape) else 1
        analysis["total_2d_images"] = num_channels * num_z
        analysis["processing_strategy"] = (
            "Multi-channel Z-stack: process each channel*Z combination"
        )

    elif "C" in dims:
        # Multi-channel: C images
        c_idx = dims.find("C")
        num_channels = shape[c_idx] if c_idx < len(shape) else 1
        analysis["total_2d_images"] = num_channels
        analysis["processing_strategy"] = (
            "Multi-channel: process each channel separately"
        )

    elif "Z" in dims:
        # Z-stack: Z images
        z_idx = dims.find("Z")
        num_z = shape[z_idx] if z_idx < len(shape) else 1
        analysis["total_2d_images"] = num_z
        analysis["processing_strategy"] = "Z-stack: process each Z-slice"

    else:
        # Single 2D image
        analysis["total_2d_images"] = 1
        analysis["processing_strategy"] = "Single 2D image"

    # Time series multiplier
    if "T" in dims:
        t_idx = dims.find("T")
        num_t = shape[t_idx] if t_idx < len(shape) else 1
        analysis["total_2d_images"] *= num_t
        analysis["processing_strategy"] += f" across {num_t} timepoints"

    analysis["estimated_iscc_codes"] = analysis["total_2d_images"]

    # Memory efficiency assessment
    if len(slices) > 0:
        sample_slice = slices[0][1]
        slice_memory = sample_slice.nbytes
        total_slice_memory = slice_memory * analysis["total_2d_images"]

        if total_slice_memory > 1e9:  # > 1GB
            analysis["memory_efficient"] = False
            analysis["processing_strategy"] += " (requires chunked processing)"

    print(f"    Total 2D images for ISCC: {analysis['total_2d_images']}")
    print(f"    Strategy: {analysis['processing_strategy']}")
    print(f"    Memory efficient: {analysis['memory_efficient']}")

    return analysis


def main():
    """Main analysis function."""
    bioimages_dir = Path("E:/biocodes/bioimages")

    print("BIOIMAGE ANALYSIS FOR ISCC PROCESSING")
    print("=" * 70)
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    # Files to analyze in detail
    detailed_files = [
        "xyc_tiles.czi",  # Multi-channel 2D
        "40xsiliconpollen.nd2",  # Multi-channel Z-stack
        "Comp.tif",  # Large single 2D image
    ]

    results = []

    # Detailed analysis
    for filename in detailed_files:
        file_path = bioimages_dir / filename
        if file_path.exists():
            result = analyze_for_iscc_processing(str(file_path))
            results.append(result)

    # Quick overview of large files
    print(f"\n{'=' * 70}")
    print("LARGE FILES OVERVIEW")
    print(f"{'=' * 70}")

    large_files = [
        "3Dfish-001.lif",
        "WT A17 Myotype 2-003.czi",
    ]

    for filename in large_files:
        file_path = bioimages_dir / filename
        if file_path.exists():
            try:
                bio_img = BioImage(str(file_path))
                file_size = os.path.getsize(file_path)

                print(f"\n{filename}: {format_size(file_size)}")
                print(f"  Shape: {bio_img.shape}")
                print(f"  Dimensions: {bio_img.dims}")

                # Quick ISCC estimate
                dims = str(bio_img.dims)
                shape = bio_img.shape

                total_2d = 1
                if "C" in dims:
                    total_2d *= shape[dims.find("C")]
                if "Z" in dims:
                    total_2d *= shape[dims.find("Z")]
                if "T" in dims:
                    total_2d *= shape[dims.find("T")]

                print(f"  Estimated 2D slices for ISCC: {total_2d}")

                if hasattr(bio_img, "channel_names") and bio_img.channel_names:
                    print(
                        f"  Channels: {[str(name) for name in bio_img.channel_names]}"
                    )

            except Exception as e:
                print(f"\n{filename}: Error - {e}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("ISCC PROCESSING SUMMARY")
    print(f"{'=' * 70}")

    successful_results = [r for r in results if "error" not in r]

    for result in successful_results:
        name = Path(result["file_path"]).name
        iscc = result["iscc_analysis"]

        print(f"\n{name}:")
        print(f"  - {iscc['total_2d_images']} individual 2D images to process")
        print(f"  - Strategy: {iscc['processing_strategy']}")
        print(f"  - Memory efficient: {iscc['memory_efficient']}")
        print(f"  - File size: {format_size(result['file_size'])}")

    print("\nRECOMMENDATIONS FOR ISCC IMPLEMENTATION:")
    print("1. Process bioimage files as collections of 2D slices")
    print("2. Generate individual ISCC Image-Codes for each 2D slice")
    print("3. Combine Image-Codes into ISCC Mixed-Code for complete bioimage")
    print("4. Use lazy loading and chunked processing for large files")
    print("5. Consider parallel processing for multi-channel/Z-stack data")


if __name__ == "__main__":
    main()
