"""Test script for the new views extraction functionality."""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iscc_bio.views import extract_views, views_to_thumbnails, ViewInfo

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_views_extraction():
    """Test the views extraction on sample bioimages."""

    # Look for sample files
    sample_dir = Path("E:/biocodes/bioimages")
    if not sample_dir.exists():
        sample_dir = Path(".")

    # Find a test file (skip problematic Comp.tif)
    test_files = (
        [f for f in sample_dir.glob("*.tif") if f.name != "Comp.tif"]
        + list(sample_dir.glob("*.tiff"))
        + list(sample_dir.glob("*.czi"))
        + list(sample_dir.glob("*.nd2"))
        + list(sample_dir.glob("*.lif"))
    )

    if not test_files:
        logger.warning("No test files found. Please provide a bioimage file.")
        return

    test_file = test_files[0]
    logger.info(f"Testing with file: {test_file}")

    # Test different strategy combinations
    strategy_sets = [
        ["mip", "best_focus", "representative"],  # Default
        ["mip"],  # MIP only
        ["best_focus"],  # Focus only
        ["representative"],  # Sampling only
        ["composite"],  # RGB composite
        ["mip", "composite"],  # Combined
    ]

    for strategies in strategy_sets:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing strategies: {strategies}")
        logger.info(f"{'=' * 60}")

        try:
            # Extract views
            views = extract_views(
                test_file, max_views=8, strategies=strategies, min_quality_score=0.1
            )

            logger.info(f"Extracted {len(views)} views:")
            for i, view in enumerate(views):
                logger.info(f"  View {i + 1}:")
                logger.info(f"    Type: {view.view_type}")
                logger.info(f"    Shape: {view.data.shape}")
                logger.info(f"    Dtype: {view.data.dtype}")
                if view.timepoint is not None:
                    logger.info(f"    Timepoint: {view.timepoint}")
                if view.z_plane is not None:
                    logger.info(f"    Z-plane: {view.z_plane}")
                if view.channels:
                    logger.info(f"    Channels: {view.channels}")
                if view.metadata:
                    logger.info(f"    Metadata: {view.metadata}")

            # Save thumbnails
            output_dir = Path("test_views") / "_".join(strategies)
            thumbnails = views_to_thumbnails(views, output_dir=output_dir)
            logger.info(f"Saved {len(thumbnails)} thumbnails to {output_dir}")

        except Exception as e:
            logger.error(f"Error with strategies {strategies}: {e}")
            import traceback

            traceback.print_exc()


def test_memory_efficiency():
    """Test memory efficiency with large files."""
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Look for a large file
    sample_dir = Path("E:/biocodes/bioimages")
    if not sample_dir.exists():
        logger.warning("Sample directory not found")
        return

    large_files = [f for f in sample_dir.glob("*") if f.stat().st_size > 100_000_000]

    if not large_files:
        logger.info("No large files found for memory testing")
        return

    test_file = large_files[0]
    logger.info(f"Testing memory efficiency with: {test_file}")
    logger.info(f"File size: {test_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Measure memory before
    mem_before = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory before: {mem_before:.1f} MB")

    # Extract views
    views = extract_views(test_file, max_views=8)

    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory after: {mem_after:.1f} MB")
    logger.info(f"Memory increase: {mem_after - mem_before:.1f} MB")
    logger.info(f"Extracted {len(views)} views")


def test_omero_compatibility():
    """Test OMERO compatibility (requires OMERO connection)."""
    try:
        from omero.gateway import BlitzGateway
    except ImportError:
        logger.info("OMERO Python not installed, skipping OMERO test")
        return

    logger.info("Testing OMERO compatibility...")
    logger.info("This would require OMERO server connection details")

    # Example OMERO usage (would need actual server):
    # conn = BlitzGateway(username, password, host=host, port=port)
    # conn.connect()
    # image = conn.getObject("Image", image_id)
    # views = extract_views(image, max_views=8)


if __name__ == "__main__":
    logger.info("Starting views extraction tests...")

    # Run basic extraction test
    test_views_extraction()

    # Test memory efficiency
    test_memory_efficiency()

    # Test OMERO compatibility (if available)
    test_omero_compatibility()

    logger.info("\nTests completed!")
