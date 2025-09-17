"""
Simple Bioimage Test Script

Direct testing of bioio functionality and data access.
"""

import os
from pathlib import Path
import numpy as np
from bioio import BioImage


def test_simple_access(file_path: str):
    """Simple test of bioimage data access."""
    print(f"\n{'='*50}")
    print(f"Testing: {Path(file_path).name}")
    print(f"{'='*50}")

    try:
        bio_img = BioImage(file_path)

        print(f"Shape: {bio_img.shape}")
        print(f"Dimensions: {bio_img.dims}")
        print(f"Data type: {bio_img.dtype}")

        # Try to access raw data directly
        print(f"\nTrying direct data access...")

        # Method 1: Direct data property
        try:
            data = bio_img.data
            print(f"  bio_img.data shape: {data.shape}")
            print(f"  bio_img.data type: {type(data)}")

            # Try simple slicing
            if len(data.shape) >= 2:
                # Get a 2D slice from the middle
                if len(data.shape) == 5:  # TCZYX
                    slice_2d = data[0, 0, 0, :, :]  # First T, C, Z, all Y, X
                elif len(data.shape) == 4:  # CZYX
                    slice_2d = data[0, 0, :, :]  # First C, Z, all Y, X
                elif len(data.shape) == 3:  # ZYX or CYX
                    slice_2d = data[0, :, :]  # First Z/C, all Y, X
                else:
                    slice_2d = data  # Already 2D

                print(f"  2D slice shape: {slice_2d.shape}")
                print(f"  2D slice stats: min={slice_2d.min()}, max={slice_2d.max()}, mean={slice_2d.mean():.2f}")

        except Exception as e:
            print(f"  Error with direct data access: {e}")

        # Method 2: Using get_image_data
        try:
            print(f"\nTrying get_image_data...")
            # Try to get just YX dimensions
            img_yx = bio_img.get_image_data("YX")
            print(f"  YX data shape: {img_yx.shape}")
            print(f"  YX stats: min={img_yx.min()}, max={img_yx.max()}, mean={img_yx.mean():.2f}")

        except Exception as e:
            print(f"  Error with get_image_data YX: {e}")

        # Method 3: Try different dimension strings
        try:
            print(f"\nTrying alternative dimension access...")
            img_cyx = bio_img.get_image_data("CYX")
            print(f"  CYX data shape: {img_cyx.shape}")

        except Exception as e:
            print(f"  Error with get_image_data CYX: {e}")

        # Check if dask array
        try:
            data_type = type(bio_img.data)
            print(f"\nData type details: {data_type}")
            if hasattr(bio_img.data, 'compute'):
                print("  Data is a dask array - supports lazy loading")
            if hasattr(bio_img.data, 'chunks'):
                print(f"  Chunks: {bio_img.data.chunks}")

        except Exception as e:
            print(f"  Error checking data type: {e}")

    except Exception as e:
        print(f"Error loading file: {e}")


def main():
    """Test bioimage files."""
    bioimages_dir = Path("E:/biocodes/bioimages")

    # Test smaller files first
    test_files = [
        "xyc_tiles.czi",
        "40xsiliconpollen.nd2",
    ]

    for filename in test_files:
        file_path = bioimages_dir / filename
        if file_path.exists():
            test_simple_access(str(file_path))
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()