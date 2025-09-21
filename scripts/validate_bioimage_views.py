"""
Visual validation tool for bioimage 2D view extraction.

This script extracts biologically meaningful 2D views from bioimages and displays them
for visual inspection before ISCC Image-Code generation.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from bioio import BioImage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any
import warnings

# Suppress bioformats warnings if present
warnings.filterwarnings("ignore", category=UserWarning)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize image array for display (0-1 range)."""
    if image.size == 0:
        return image

    # Handle different data types
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    else:
        # General normalization
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image.astype(np.float32)


def extract_mip_views(bio_img: BioImage) -> List[Dict[str, Any]]:
    """Extract maximum intensity projections for each channel and timepoint."""
    views = []

    # Skip if no Z-stack
    if bio_img.dims.Z <= 1:
        return views

    print(f"  Extracting MIPs from {bio_img.dims.Z} Z-planes...")

    for t in range(bio_img.dims.T):
        for c in range(bio_img.dims.C):
            # Get all Z-slices for this channel/timepoint
            z_stack = bio_img.get_image_data("ZYX", T=t, C=c)

            # Create MIP - collapse Z dimension by taking maximum
            mip_2d = np.max(z_stack, axis=0)  # Results in YX

            channel_name = (
                bio_img.channel_names[c] if c < len(bio_img.channel_names) else f"Ch{c}"
            )

            views.append(
                {
                    "data": mip_2d,
                    "type": "MIP",
                    "title": f"MIP - T{t} - {channel_name}",
                    "channel": c,
                    "channel_name": channel_name,
                    "timepoint": t,
                }
            )

    return views


def extract_composite_views(
    bio_img: BioImage, use_mip: bool = True
) -> List[Dict[str, Any]]:
    """Create composite images from multiple channels."""
    views = []

    # Skip if single channel
    if bio_img.dims.C <= 1:
        return views

    print(f"  Creating composite views from {bio_img.dims.C} channels...")

    for t in range(bio_img.dims.T):
        if use_mip and bio_img.dims.Z > 1:
            # Use MIP for composite
            channels = []
            for c in range(min(3, bio_img.dims.C)):  # Max 3 channels for RGB
                z_stack = bio_img.get_image_data("ZYX", T=t, C=c)
                mip = np.max(z_stack, axis=0)
                channels.append(normalize_for_display(mip))

            # Pad with zeros if less than 3 channels
            while len(channels) < 3:
                channels.append(np.zeros_like(channels[0]))

            composite = np.stack(channels, axis=-1)

            views.append(
                {
                    "data": composite,
                    "type": "Composite",
                    "title": f"Composite MIP - T{t}",
                    "timepoint": t,
                    "is_rgb": True,
                }
            )
        else:
            # Use single Z-plane for composite
            for z in range(
                min(3, bio_img.dims.Z)
            ):  # Limit to first 3 Z-planes for display
                channels = []
                for c in range(min(3, bio_img.dims.C)):
                    img = bio_img.get_image_data("YX", T=t, C=c, Z=z)
                    channels.append(normalize_for_display(img))

                # Pad with zeros if less than 3 channels
                while len(channels) < 3:
                    channels.append(np.zeros_like(channels[0]))

                composite = np.stack(channels, axis=-1)

                views.append(
                    {
                        "data": composite,
                        "type": "Composite",
                        "title": f"Composite - T{t} Z{z}",
                        "z_plane": z,
                        "timepoint": t,
                        "is_rgb": True,
                    }
                )

    return views


def extract_best_focus_views(bio_img: BioImage) -> List[Dict[str, Any]]:
    """Extract best-focused planes from Z-stacks."""
    views = []

    # Skip if no Z-stack
    if bio_img.dims.Z <= 1:
        return views

    print(f"  Finding best focus planes...")

    for t in range(bio_img.dims.T):
        for c in range(bio_img.dims.C):
            z_stack = bio_img.get_image_data("ZYX", T=t, C=c)

            # Find sharpest Z-plane using variance of Laplacian
            sharpness_scores = []
            for z in range(len(z_stack)):
                # Variance of Laplacian as focus measure
                laplacian = np.gradient(np.gradient(z_stack[z], axis=0), axis=1)
                score = np.var(laplacian)
                sharpness_scores.append(score)

            best_z = np.argmax(sharpness_scores)
            channel_name = (
                bio_img.channel_names[c] if c < len(bio_img.channel_names) else f"Ch{c}"
            )

            views.append(
                {
                    "data": z_stack[best_z],
                    "type": "Best Focus",
                    "title": f"Best Focus Z{best_z} - T{t} - {channel_name}",
                    "z_index": best_z,
                    "channel": c,
                    "channel_name": channel_name,
                    "timepoint": t,
                }
            )

    return views


def extract_sample_planes(
    bio_img: BioImage, max_samples: int = 6
) -> List[Dict[str, Any]]:
    """Extract sample individual planes for inspection."""
    views = []

    print(f"  Extracting sample planes...")

    # Calculate sampling interval
    z_samples = min(max_samples, bio_img.dims.Z)
    z_indices = (
        np.linspace(0, bio_img.dims.Z - 1, z_samples, dtype=int)
        if bio_img.dims.Z > 1
        else [0]
    )

    for t in range(min(1, bio_img.dims.T)):  # Just first timepoint for samples
        for c in range(min(2, bio_img.dims.C)):  # Just first 2 channels for samples
            for z in z_indices[:3]:  # Max 3 Z samples per channel
                img = bio_img.get_image_data("YX", T=t, C=c, Z=z)
                channel_name = (
                    bio_img.channel_names[c]
                    if c < len(bio_img.channel_names)
                    else f"Ch{c}"
                )

                views.append(
                    {
                        "data": img,
                        "type": "Single Plane",
                        "title": f"Plane - T{t} {channel_name} Z{z}",
                        "coords": {"T": t, "C": c, "Z": z},
                        "channel_name": channel_name,
                    }
                )

    return views


def extract_biological_views(
    bio_img: BioImage, strategy: str = "comprehensive"
) -> List[Dict[str, Any]]:
    """
    Extract biologically meaningful 2D views from a bioimage.

    Strategies:
    - comprehensive: All view types (MIP, composite, best focus, samples)
    - mip_only: Only maximum intensity projections
    - composite_only: Only multi-channel composites
    - samples: Individual plane samples
    """
    views = []

    print(f"\nExtracting {strategy} views...")
    print(
        f"  Image dimensions: T={bio_img.dims.T}, C={bio_img.dims.C}, "
        f"Z={bio_img.dims.Z}, Y={bio_img.dims.Y}, X={bio_img.dims.X}"
    )

    if strategy in ["comprehensive", "mip_only"]:
        views.extend(extract_mip_views(bio_img))

    if strategy in ["comprehensive", "composite_only"]:
        views.extend(extract_composite_views(bio_img))

    if strategy == "comprehensive" and bio_img.dims.Z > 1:
        views.extend(extract_best_focus_views(bio_img))

    if strategy in ["comprehensive", "samples"]:
        views.extend(extract_sample_planes(bio_img))

    # Sort for deterministic ordering
    views.sort(
        key=lambda v: (
            v["type"],
            v.get("timepoint", 0),
            v.get("channel", 0),
            v.get("z_plane", v.get("z_index", 0)),
        )
    )

    return views


def display_views(views: List[Dict[str, Any]], save_path: Path = None):
    """Display extracted views in a grid layout."""
    if not views:
        print("No views to display!")
        return

    # Group views by type
    view_types = {}
    for view in views:
        view_type = view["type"]
        if view_type not in view_types:
            view_types[view_type] = []
        view_types[view_type].append(view)

    # Create figure
    total_views = len(views)
    cols = min(4, total_views)
    rows = (total_views + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4, rows * 4))
    fig.suptitle(
        f"Bioimage View Extraction ({total_views} views)",
        fontsize=16,
        fontweight="bold",
    )

    # Plot each view
    for idx, view in enumerate(views):
        ax = fig.add_subplot(rows, cols, idx + 1)

        img_data = view["data"]

        # Handle RGB vs grayscale
        if view.get("is_rgb", False) or (img_data.ndim == 3 and img_data.shape[2] == 3):
            ax.imshow(img_data)
        else:
            # Grayscale
            img_display = normalize_for_display(img_data)
            ax.imshow(img_display, cmap="gray")

        ax.set_title(view["title"], fontsize=10, fontweight="bold")
        ax.axis("off")

        # Add info text
        info = f"Shape: {img_data.shape}"
        if "z_index" in view:
            info += f"\nZ: {view['z_index']}"
        ax.text(
            0.02,
            0.02,
            info,
            transform=ax.transAxes,
            fontsize=8,
            color="yellow",
            backgroundcolor="black",
            alpha=0.7,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nViews saved to: {save_path}")

    plt.show()

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    for view_type, type_views in view_types.items():
        print(f"{view_type:15} : {len(type_views)} views")
    print(f"{'=' * 60}")
    print(f"{'Total':15} : {total_views} views")
    print(f"{'=' * 60}")

    print(f"\nThese {total_views} views would each generate an ISCC Image-Code,")
    print(f"which would be combined into a single Mixed-Code for the bioimage.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate bioimage 2D view extraction for ISCC generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_bioimage_views.py E:/biocodes/bioimages/xyc_tiles.czi
  python validate_bioimage_views.py E:/biocodes/bioimages/40xsiliconpollen.nd2 --strategy mip_only
  python validate_bioimage_views.py bioimage.tif --save output.png
        """,
    )

    parser.add_argument("filepath", type=str, help="Path to the bioimage file")

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["comprehensive", "mip_only", "composite_only", "samples"],
        default="comprehensive",
        help="View extraction strategy (default: comprehensive)",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save visualization to file (e.g., output.png)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display the plot (useful for headless environments)",
    )

    args = parser.parse_args()

    # Check if file exists
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Loading bioimage: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    try:
        # Load bioimage
        bio_img = BioImage(filepath)

        # Print metadata
        print(f"\nImage Information:")
        print(f"  Format: {filepath.suffix}")
        print(f"  Dimensions: {bio_img.dims.order}")
        print(f"  Shape: {bio_img.shape}")
        print(f"  Data type: {bio_img.dtype}")

        if bio_img.channel_names:
            print(f"  Channels: {', '.join(bio_img.channel_names)}")

        if bio_img.physical_pixel_sizes.Z or bio_img.physical_pixel_sizes.Y:
            print(
                f"  Pixel sizes: Z={bio_img.physical_pixel_sizes.Z}μm, "
                f"Y={bio_img.physical_pixel_sizes.Y}μm, "
                f"X={bio_img.physical_pixel_sizes.X}μm"
            )

        # Extract views
        views = extract_biological_views(bio_img, strategy=args.strategy)

        if not views:
            print("\nNo views could be extracted from this image!")
            sys.exit(1)

        # Display views
        save_path = Path(args.save) if args.save else None

        if not args.no_display or save_path:
            display_views(views, save_path)
        else:
            print(f"\nExtracted {len(views)} views (display skipped)")

    except Exception as e:
        print(f"\nError processing bioimage: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
