"""
Bioimage Analysis Script v2

Simplified script to analyze bioimage files using bioio for ISCC processing planning.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import psutil
import numpy as np

try:
    from bioio import BioImage
except ImportError:
    print("Error: bioio not installed. Run 'uv sync --dev' to install dependencies.")
    sys.exit(1)


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def analyze_bioimage_simple(file_path: str) -> Dict[str, Any]:
    """
    Analyze a bioimage file - simpler version focusing on basic properties.
    """
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {Path(file_path).name}")
    print(f"File size: {format_size(os.path.getsize(file_path))}")
    print(f"{'=' * 60}")

    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} GB")

    try:
        # Load bioimage
        bio_img = BioImage(file_path)

        # Basic information
        shape = bio_img.shape
        dims = bio_img.dims
        dtype = bio_img.dtype

        print(f"Shape: {shape}")
        print(f"Dimensions: {dims}")
        print(f"Data type: {dtype}")

        # Calculate memory requirements
        total_pixels = np.prod(shape)
        bytes_per_pixel = dtype.itemsize
        total_bytes = total_pixels * bytes_per_pixel
        estimated_memory = format_size(total_bytes)
        print(f"Estimated memory for full load: {estimated_memory}")

        # Physical pixel sizes
        if bio_img.physical_pixel_sizes:
            pps = bio_img.physical_pixel_sizes
            print("Physical pixel sizes:")
            if hasattr(pps, "Z") and pps.Z is not None:
                print(f"  Z: {pps.Z:.4f}")
            if hasattr(pps, "Y") and pps.Y is not None:
                print(f"  Y: {pps.Y:.4f}")
            if hasattr(pps, "X") and pps.X is not None:
                print(f"  X: {pps.X:.4f}")

        # Channel information
        try:
            if hasattr(bio_img, "channel_names") and bio_img.channel_names:
                print(f"Channel names: {bio_img.channel_names}")
        except Exception as e:
            print(f"Could not get channel names: {e}")

        # Scene information
        try:
            if hasattr(bio_img, "scenes") and bio_img.scenes:
                print(f"Number of scenes: {len(bio_img.scenes)}")
        except Exception as e:
            print(f"Could not get scene info: {e}")

        # Extract a simple 2D slice using bioio's built-in methods
        print("\nExtracting sample 2D slice...")
        try:
            memory_before = get_memory_usage()

            # For 2D extraction, we want to get a single XY plane
            # Let's try to get the middle slice if there are Z/T dimensions
            slice_kwargs = {}

            # Parse dimensions to understand the structure
            dim_names = [str(d) for d in dims]
            print(f"Dimension names: {dim_names}")

            # Handle different dimension patterns
            if "T" in dim_names and shape[dim_names.index("T")] > 1:
                slice_kwargs["T"] = shape[dim_names.index("T")] // 2
            if "Z" in dim_names and shape[dim_names.index("Z")] > 1:
                slice_kwargs["Z"] = shape[dim_names.index("Z")] // 2
            if "C" in dim_names:
                slice_kwargs["C"] = 0  # First channel

            print(f"Using slice parameters: {slice_kwargs}")

            # Get the 2D slice
            slice_data = bio_img.get_image_data("YX", **slice_kwargs)

            memory_after = get_memory_usage()

            print(f"Extracted 2D slice shape: {slice_data.shape}")
            print(f"Slice dtype: {slice_data.dtype}")
            print(f"Memory used for slice: {memory_after - memory_before:.3f} GB")
            print(
                f"Slice stats: min={slice_data.min()}, max={slice_data.max()}, mean={slice_data.mean():.2f}"
            )

            # If multi-channel, try a few channels
            if "C" in dim_names and shape[dim_names.index("C")] > 1:
                num_channels = shape[dim_names.index("C")]
                print(f"\nMulti-channel image with {num_channels} channels:")

                for c in range(min(3, num_channels)):  # Max 3 channels
                    try:
                        channel_kwargs = slice_kwargs.copy()
                        channel_kwargs["C"] = c
                        channel_slice = bio_img.get_image_data("YX", **channel_kwargs)
                        print(
                            f"  Channel {c}: shape={channel_slice.shape}, "
                            f"min={channel_slice.min()}, max={channel_slice.max()}, "
                            f"mean={channel_slice.mean():.2f}"
                        )
                    except Exception as e:
                        print(f"  Channel {c}: Error - {e}")

        except Exception as e:
            print(f"Error extracting slice: {e}")
            import traceback

            traceback.print_exc()

        final_memory = get_memory_usage()
        print(
            f"\nMemory usage: {initial_memory:.2f} GB -> {final_memory:.2f} GB "
            f"(+{final_memory - initial_memory:.2f} GB)"
        )

        return {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "shape": shape,
            "dims": str(dims),
            "dtype": str(dtype),
            "estimated_memory": estimated_memory,
            "memory_used": final_memory - initial_memory,
        }

    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "file_path": file_path}


def main():
    """Main analysis function."""
    bioimages_dir = Path("E:/biocodes/bioimages")

    if not bioimages_dir.exists():
        print(f"Bioimages directory not found: {bioimages_dir}")
        return

    # Files to analyze in order of size
    files_to_analyze = [
        "xyc_tiles.czi",  # ~20MB - Small multi-channel
        "40xsiliconpollen.nd2",  # ~462MB - Z-stack
        "Comp.tif",  # ~540MB - Large 2D composite
    ]

    print("Starting bioimage analysis...")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    results = []

    for filename in files_to_analyze:
        file_path = bioimages_dir / filename

        if file_path.exists():
            try:
                result = analyze_bioimage_simple(str(file_path))
                results.append(result)

                # Small pause
                import time

                time.sleep(1)

            except KeyboardInterrupt:
                print("\nAnalysis interrupted by user")
                break
            except Exception as e:
                print(f"Failed to analyze {filename}: {e}")

        else:
            print(f"File not found: {file_path}")

    # Brief look at larger files (metadata only)
    print(f"\n{'=' * 60}")
    print("LARGE FILES (metadata only)")
    print(f"{'=' * 60}")

    large_files = [
        "3Dfish-001.lif",
        "WT A17 Myotype 2-003.czi",
    ]

    for filename in large_files:
        file_path = bioimages_dir / filename
        if file_path.exists():
            file_size = os.path.getsize(file_path)
            print(f"\n{filename}: {format_size(file_size)}")

            try:
                bio_img = BioImage(str(file_path))
                shape = bio_img.shape
                dims = bio_img.dims
                dtype = bio_img.dtype

                total_pixels = np.prod(shape)
                bytes_per_pixel = dtype.itemsize
                total_bytes = total_pixels * bytes_per_pixel
                estimated_memory = format_size(total_bytes)

                print(f"  Shape: {shape}")
                print(f"  Dimensions: {dims}")
                print(f"  Data type: {dtype}")
                print(f"  Estimated memory: {estimated_memory}")

                # Channel info if available
                if hasattr(bio_img, "channel_names") and bio_img.channel_names:
                    print(f"  Channels: {bio_img.channel_names}")

            except Exception as e:
                print(f"  Error: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 60}")

    successful_results = [r for r in results if "error" not in r]

    if successful_results:
        total_files = len(successful_results)
        avg_memory = sum(r["memory_used"] for r in successful_results) / total_files

        print(f"Successfully analyzed {total_files} files")
        print(f"Average memory usage per file: {avg_memory:.3f} GB")
        print("\nFile types and structures found:")

        for result in successful_results:
            name = Path(result["file_path"]).name
            print(f"  {name}: {result['dims']} -> {result['shape']}")

        print("\nISCC Processing Recommendations:")
        print("1. Use lazy loading for files > 500MB")
        print("2. Process 2D slices individually to manage memory")
        print("3. Consider chunked processing for large Z-stacks")
        print("4. Multi-channel images: process each channel separately")


if __name__ == "__main__":
    main()
