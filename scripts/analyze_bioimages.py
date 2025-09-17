"""
Bioimage Analysis Script

This script analyzes bioimage files using bioio to understand their structure,
dimensions, and properties for ISCC processing planning.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import psutil
import numpy as np

try:
    from bioio import BioImage
    from bioio_base import types
except ImportError:
    print("Error: bioio not installed. Run 'uv sync --dev' to install dependencies.")
    sys.exit(1)


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def estimate_memory_requirements(shape: Tuple[int, ...], dtype: np.dtype) -> str:
    """Estimate memory requirements for loading full image."""
    total_pixels = np.prod(shape)
    bytes_per_pixel = dtype.itemsize
    total_bytes = total_pixels * bytes_per_pixel
    return format_size(total_bytes)


def analyze_bioimage(file_path: str, extract_slices: bool = True) -> Dict[str, Any]:
    """
    Analyze a bioimage file and extract key information.

    Args:
        file_path: Path to the bioimage file
        extract_slices: Whether to extract sample 2D slices

    Returns:
        Dictionary containing analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(file_path).name}")
    print(f"File size: {format_size(os.path.getsize(file_path))}")
    print(f"{'='*60}")

    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} GB")

    try:
        # Load bioimage with lazy loading
        bio_img = BioImage(file_path)

        # Get basic information
        analysis = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'shape': bio_img.shape,
            'dtype': bio_img.dtype,
            'dimensions': bio_img.dims,
            'physical_pixel_sizes': bio_img.physical_pixel_sizes,
            'metadata': None,
            'channel_names': None,
            'scenes': None,
            'estimated_memory': estimate_memory_requirements(bio_img.shape, bio_img.dtype)
        }

        print(f"Shape: {bio_img.shape}")
        print(f"Dimensions: {bio_img.dims}")
        print(f"Data type: {bio_img.dtype}")
        print(f"Estimated memory for full load: {analysis['estimated_memory']}")

        # Get physical pixel sizes
        if bio_img.physical_pixel_sizes:
            pps = bio_img.physical_pixel_sizes
            print(f"Physical pixel sizes:")
            if pps.Z is not None:
                print(f"  Z: {pps.Z:.4f} {pps.Z}")
            if pps.Y is not None:
                print(f"  Y: {pps.Y:.4f}")
            if pps.X is not None:
                print(f"  X: {pps.X:.4f}")

        # Try to get metadata
        try:
            metadata = bio_img.metadata
            analysis['metadata'] = str(type(metadata))
            print(f"Metadata type: {type(metadata)}")
        except Exception as e:
            print(f"Could not access metadata: {e}")

        # Try to get channel names
        try:
            if hasattr(bio_img, 'channel_names') and bio_img.channel_names:
                analysis['channel_names'] = bio_img.channel_names
                print(f"Channel names: {bio_img.channel_names}")
        except Exception as e:
            print(f"Could not access channel names: {e}")

        # Check for multiple scenes
        try:
            if hasattr(bio_img, 'scenes') and bio_img.scenes:
                analysis['scenes'] = len(bio_img.scenes)
                print(f"Number of scenes: {len(bio_img.scenes)}")
        except Exception as e:
            print(f"Could not access scenes: {e}")

        # Extract sample 2D slices if requested
        if extract_slices:
            print("\nExtracting sample 2D slices...")
            try:
                # Get dimensions
                dims = bio_img.dims
                shape = bio_img.shape

                # Find indices for different dimensions
                dims_str = str(dims)
                t_idx = dims_str.find('T') if 'T' in dims_str else None
                z_idx = dims_str.find('Z') if 'Z' in dims_str else None
                c_idx = dims_str.find('C') if 'C' in dims_str else None
                y_idx = dims_str.find('Y') if 'Y' in dims_str else None
                x_idx = dims_str.find('X') if 'X' in dims_str else None

                # Get actual indices using dimension order
                dim_order = [d for d in dims_str if d in 'TCZYX']
                t_idx = dim_order.index('T') if 'T' in dim_order else None
                z_idx = dim_order.index('Z') if 'Z' in dim_order else None
                c_idx = dim_order.index('C') if 'C' in dim_order else None
                y_idx = dim_order.index('Y') if 'Y' in dim_order else None
                x_idx = dim_order.index('X') if 'X' in dim_order else None

                print(f"Dimension positions: T={t_idx}, Z={z_idx}, C={c_idx}, Y={y_idx}, X={x_idx}")

                # Create slice for middle of volume/time series
                slice_coords = [0] * len(shape)

                if t_idx is not None and shape[t_idx] > 1:
                    slice_coords[t_idx] = shape[t_idx] // 2
                if z_idx is not None and shape[z_idx] > 1:
                    slice_coords[z_idx] = shape[z_idx] // 2
                if c_idx is not None:
                    slice_coords[c_idx] = 0  # First channel

                # Convert to tuple for slicing
                slice_tuple = tuple(slice_coords)

                print(f"Extracting slice at coordinates: {slice_tuple}")

                # Extract the slice
                memory_before = get_memory_usage()
                slice_data = bio_img.data[slice_tuple]
                memory_after = get_memory_usage()

                print(f"Slice shape: {slice_data.shape}")
                print(f"Slice dtype: {slice_data.dtype}")
                print(f"Slice memory usage: {memory_after - memory_before:.3f} GB")
                print(f"Slice stats: min={slice_data.min()}, max={slice_data.max()}, mean={slice_data.mean():.2f}")

                # If there are multiple channels, extract first few
                if c_idx is not None and shape[c_idx] > 1:
                    num_channels_to_check = min(3, shape[c_idx])
                    print(f"\nChecking first {num_channels_to_check} channels:")

                    for c in range(num_channels_to_check):
                        channel_coords = list(slice_coords)
                        channel_coords[c_idx] = c
                        channel_slice = bio_img.data[tuple(channel_coords)]
                        print(f"  Channel {c}: shape={channel_slice.shape}, "
                              f"stats: min={channel_slice.min()}, max={channel_slice.max()}, "
                              f"mean={channel_slice.mean():.2f}")

            except Exception as e:
                print(f"Error extracting slices: {e}")
                import traceback
                traceback.print_exc()

        final_memory = get_memory_usage()
        print(f"\nMemory usage: {initial_memory:.2f} GB -> {final_memory:.2f} GB "
              f"(+{final_memory - initial_memory:.2f} GB)")

        return analysis

    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'file_path': file_path}


def main():
    """Main analysis function."""
    bioimages_dir = Path("E:/biocodes/bioimages")

    if not bioimages_dir.exists():
        print(f"Bioimages directory not found: {bioimages_dir}")
        return

    # Define files to analyze in order of increasing size
    files_to_analyze = [
        "xyc_tiles.czi",           # ~20MB
        "human kidney_0001.oir",   # ~98MB
        "40xsiliconpollen.nd2",    # ~462MB
        "Comp.tif",                # ~540MB
    ]

    results = []

    print(f"Starting bioimage analysis...")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    for filename in files_to_analyze:
        file_path = bioimages_dir / filename

        if file_path.exists():
            try:
                result = analyze_bioimage(str(file_path))
                results.append(result)

                # Brief pause to see memory usage
                import time
                time.sleep(2)

            except KeyboardInterrupt:
                print("\nAnalysis interrupted by user")
                break
            except Exception as e:
                print(f"Failed to analyze {filename}: {e}")
                results.append({'error': str(e), 'file_path': str(file_path)})
        else:
            print(f"File not found: {file_path}")

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    for i, result in enumerate(results):
        if 'error' not in result:
            print(f"{i+1}. {Path(result['file_path']).name}")
            print(f"   Shape: {result['shape']}")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   File size: {format_size(result['file_size'])}")
            print(f"   Est. memory: {result['estimated_memory']}")
        else:
            print(f"{i+1}. {Path(result['file_path']).name} - ERROR: {result['error']}")

    # Quick look at larger files (without full analysis)
    print(f"\n{'='*60}")
    print("LARGE FILES (quick inspection)")
    print(f"{'='*60}")

    large_files = [
        "3Dfish-001.lif",
        "WT A17 Myotype 2-003.czi",
        "240520 EGFP-eLis1_mScarlet-eRab5_4_2024-05-20_12.51.45-002.ims"
    ]

    for filename in large_files:
        file_path = bioimages_dir / filename
        if file_path.exists():
            file_size = os.path.getsize(file_path)
            print(f"{filename}: {format_size(file_size)}")

            try:
                # Quick metadata-only inspection
                bio_img = BioImage(str(file_path))
                print(f"  Shape: {bio_img.shape}")
                print(f"  Dimensions: {bio_img.dims}")
                print(f"  Estimated memory: {estimate_memory_requirements(bio_img.shape, bio_img.dtype)}")
            except Exception as e:
                print(f"  Error getting basic info: {e}")


if __name__ == "__main__":
    main()