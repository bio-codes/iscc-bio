"""
Quick test script to compare different extraction strategies for ISCC generation.
"""

import sys
from pathlib import Path
from bioio import BioImage


def count_views_by_strategy(filepath: str):
    """Count how many views each strategy would generate."""

    bio_img = BioImage(filepath)

    print(f"\nFile: {Path(filepath).name}")
    print(f"Dimensions: T={bio_img.dims.T}, C={bio_img.dims.C}, Z={bio_img.dims.Z}")
    print(f"Total voxels: {bio_img.dims.T * bio_img.dims.C * bio_img.dims.Z:,}")
    print("\nView counts by strategy:")
    print("-" * 50)

    # Strategy 1: Every single plane (traditional approach)
    all_planes = bio_img.dims.T * bio_img.dims.C * bio_img.dims.Z
    print(f"All planes:        {all_planes:4d} views")

    # Strategy 2: MIP only (if Z > 1)
    if bio_img.dims.Z > 1:
        mip_only = bio_img.dims.T * bio_img.dims.C
        print(f"MIP only:          {mip_only:4d} views")

    # Strategy 3: Best focus (one per channel/timepoint)
    best_focus = bio_img.dims.T * bio_img.dims.C
    print(f"Best focus:        {best_focus:4d} views")

    # Strategy 4: Composite + MIP (recommended)
    composite_views = bio_img.dims.T  # One composite per timepoint
    if bio_img.dims.Z > 1:
        mip_views = bio_img.dims.T * bio_img.dims.C
        recommended = composite_views + mip_views
    else:
        recommended = composite_views + (bio_img.dims.T * bio_img.dims.C)
    print(f"Composite + MIP:   {recommended:4d} views")

    # Show reduction ratio
    reduction = (1 - recommended / all_planes) * 100
    print(f"\nRecommended approach reduces views by {reduction:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extraction_strategies.py <bioimage_file>")
        sys.exit(1)

    count_views_by_strategy(sys.argv[1])
