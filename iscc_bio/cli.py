"""CLI interface for iscc-bio."""

import click
from pathlib import Path
import sys
import logging
from iscc_bio.thumb import extract_thumbnail
from iscc_bio.scene import extract_scenes
from iscc_bio.views import extract_views, views_to_thumbnails
from iscc_bio.pixhash import pixhash_bioio, pixhash_omero, pixhash_zarr
from iscc_bio.biocode import generate_biocode, format_output


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """ISCC Bio - Bioimage processing tools."""
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
def thumb(input):
    """
    Extract thumbnail from bioimage file or directory.

    If INPUT is a file, creates a single thumbnail.
    If INPUT is a directory, creates thumbnails for all files in it (non-recursive).
    """
    input_path = Path(input)

    if input_path.is_file():
        # Process single file
        try:
            logger.info(f"Processing file: {input_path}")
            output_path = extract_thumbnail(input_path)
            click.echo(f" Thumbnail saved: {output_path}")
        except Exception as e:
            click.echo(f" Error processing {input_path}: {e}", err=True)
            sys.exit(1)

    elif input_path.is_dir():
        # Process all files in directory (non-recursive)
        files = [f for f in input_path.iterdir() if f.is_file()]

        if not files:
            click.echo(f"No files found in directory: {input_path}")
            sys.exit(1)

        logger.info(f"Processing {len(files)} files in directory: {input_path}")
        success_count = 0
        error_count = 0

        for file_path in files:
            # Skip already generated thumbnails
            if ".thumb.png" in file_path.name:
                logger.debug(f"Skipping thumbnail file: {file_path.name}")
                continue

            try:
                logger.info(f"Processing file: {file_path.name}")
                output_path = extract_thumbnail(file_path)
                click.echo(f" {file_path.name} -> {output_path.name}")
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                click.echo(f" {file_path.name}: {e}", err=True)
                error_count += 1

        # Summary
        click.echo(f"\nCompleted: {success_count} successful, {error_count} failed")

        if error_count > 0:
            sys.exit(1)
    else:
        click.echo(f"Error: {input_path} is neither a file nor a directory", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
def scenes(input):
    """
    Extract thumbnails from all scenes in bioimage file or directory.

    If INPUT is a file, creates thumbnails for all scenes in it.
    If INPUT is a directory, processes all files and extracts all scenes (non-recursive).
    """
    input_path = Path(input)

    if input_path.is_file():
        # Process single file
        try:
            logger.info(f"Processing file: {input_path}")
            thumbnail_paths = extract_scenes(input_path)

            if thumbnail_paths:
                click.echo(f"✓ Extracted {len(thumbnail_paths)} scene thumbnails:")
                for thumb_path in thumbnail_paths:
                    click.echo(f"  → {thumb_path.name}")
            else:
                click.echo(f"⚠ No scenes found in {input_path.name}")
        except Exception as e:
            click.echo(f"✗ Error processing {input_path}: {e}", err=True)
            sys.exit(1)

    elif input_path.is_dir():
        # Process all files in directory (non-recursive)
        files = [f for f in input_path.iterdir() if f.is_file()]

        if not files:
            click.echo(f"No files found in directory: {input_path}")
            sys.exit(1)

        logger.info(f"Processing {len(files)} files in directory: {input_path}")
        total_scenes = 0
        success_count = 0
        error_count = 0

        for file_path in files:
            # Skip already generated thumbnails
            if ".thumb.png" in file_path.name:
                logger.debug(f"Skipping thumbnail file: {file_path.name}")
                continue

            try:
                logger.info(f"Processing file: {file_path.name}")
                thumbnail_paths = extract_scenes(file_path)

                if thumbnail_paths:
                    click.echo(f"✓ {file_path.name} -> {len(thumbnail_paths)} scenes")
                    for thumb_path in thumbnail_paths:
                        click.echo(f"  → {thumb_path.name}")
                    total_scenes += len(thumbnail_paths)
                    success_count += 1
                else:
                    click.echo(f"⚠ {file_path.name}: No scenes found")
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                click.echo(f"✗ {file_path.name}: {e}", err=True)
                error_count += 1

        # Summary
        click.echo(
            f"\nCompleted: {success_count} files processed, {total_scenes} scenes extracted, {error_count} failed"
        )

        if error_count > 0:
            sys.exit(1)
    else:
        click.echo(f"Error: {input_path} is neither a file nor a directory", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--strategies",
    "-s",
    multiple=True,
    default=["mip", "best_focus", "representative"],
    help="View extraction strategies (mip, best_focus, representative, composite)",
)
@click.option(
    "--max-views",
    "-n",
    default=8,
    type=int,
    help="Maximum number of views to extract (default: 8)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save view thumbnails",
)
@click.option(
    "--host",
    help="OMERO server hostname (e.g., omero.server.com)",
)
@click.option(
    "--iid",
    type=int,
    help="OMERO image ID to process",
)
def views(input, strategies, max_views, output_dir, host, iid):
    """
    Extract intelligent views from bioimage for perceptual hashing.

    Extracts up to 8 representative 2D views using various strategies:
    - mip: Maximum intensity projections
    - best_focus: Best focus Z-planes
    - representative: Representative sampling
    - composite: Multi-channel composites

    Works efficiently with both local files and OMERO remote access.
    """
    # Check if this is OMERO mode or file mode
    if host and iid:
        # OMERO mode
        try:
            from omero.gateway import BlitzGateway

            # Connect to OMERO server with hardcoded credentials
            logger.info(f"Connecting to OMERO server: {host} on port 4064")
            conn = BlitzGateway("root", "omero", host=host, port=4064)

            if not conn.connect():
                click.echo(f"✗ Failed to connect to OMERO server: {host}", err=True)
                sys.exit(1)

            logger.info(f"Connected. Loading image ID: {iid}")
            image = conn.getObject("Image", iid)

            if not image:
                click.echo(f"✗ Image with ID {iid} not found on server", err=True)
                conn.close()
                sys.exit(1)

            try:
                click.echo(f"✓ Connected to OMERO: {image.getName()} (ID: {iid})")
                logger.info(f"Extracting views from OMERO image: {image.getName()}")
                logger.info(f"Strategies: {list(strategies)}, Max views: {max_views}")

                # Extract views from OMERO image
                extracted_views = extract_views(
                    image, max_views=max_views, strategies=list(strategies)
                )

                click.echo(f"✓ Extracted {len(extracted_views)} views:")
                for i, view in enumerate(extracted_views):
                    meta = []
                    if view.timepoint is not None:
                        meta.append(f"t={view.timepoint}")
                    if view.z_plane is not None:
                        meta.append(f"z={view.z_plane}")
                    if view.channels:
                        meta.append(f"c={view.channels}")
                    meta_str = ", ".join(meta) if meta else "default"
                    click.echo(f"  {i + 1}. {view.view_type} ({meta_str})")

                # Save thumbnails if output directory specified
                if output_dir:
                    output_dir = Path(output_dir)
                    thumbnail_paths = views_to_thumbnails(
                        extracted_views, output_dir=output_dir
                    )
                    click.echo(
                        f"\n✓ Saved {len(thumbnail_paths)} thumbnails to {output_dir}"
                    )

            finally:
                conn.close()
                logger.info("Closed OMERO connection")

        except ImportError:
            click.echo(
                "✗ OMERO Python library (omero-py) is not installed. "
                "Install it with: pip install omero-py",
                err=True,
            )
            sys.exit(1)
        except Exception as e:
            click.echo(f"✗ Error processing OMERO image: {e}", err=True)
            if "conn" in locals() and conn.isConnected():
                conn.close()
            sys.exit(1)

    elif input:
        # File mode
        input_path = Path(input)

        if input_path.is_file():
            try:
                logger.info(f"Extracting views from: {input_path}")
                logger.info(f"Strategies: {list(strategies)}, Max views: {max_views}")

                # Extract views
                extracted_views = extract_views(
                    input_path, max_views=max_views, strategies=list(strategies)
                )

                click.echo(f"✓ Extracted {len(extracted_views)} views:")
                for i, view in enumerate(extracted_views):
                    meta = []
                    if view.timepoint is not None:
                        meta.append(f"t={view.timepoint}")
                    if view.z_plane is not None:
                        meta.append(f"z={view.z_plane}")
                    if view.channels:
                        meta.append(f"c={view.channels}")
                    meta_str = ", ".join(meta) if meta else "default"
                    click.echo(f"  {i + 1}. {view.view_type} ({meta_str})")

                # Save thumbnails if output directory specified
                if output_dir:
                    output_dir = Path(output_dir)
                    thumbnail_paths = views_to_thumbnails(
                        extracted_views, output_dir=output_dir
                    )
                    click.echo(
                        f"\n✓ Saved {len(thumbnail_paths)} thumbnails to {output_dir}"
                    )

            except Exception as e:
                click.echo(f"✗ Error processing {input_path}: {e}", err=True)
                sys.exit(1)

        elif input_path.is_dir():
            # Process all files in directory
            files = [f for f in input_path.iterdir() if f.is_file()]

            if not files:
                click.echo(f"No files found in directory: {input_path}")
                sys.exit(1)

            logger.info(f"Processing {len(files)} files in directory: {input_path}")
            total_views = 0
            success_count = 0
            error_count = 0

            for file_path in files:
                # Skip thumbnails and non-bioimage files
                if ".thumb.png" in file_path.name or file_path.suffix == ".png":
                    continue

                try:
                    logger.info(f"Processing file: {file_path.name}")
                    extracted_views = extract_views(
                        file_path, max_views=max_views, strategies=list(strategies)
                    )

                    click.echo(f"✓ {file_path.name} -> {len(extracted_views)} views")
                    total_views += len(extracted_views)
                    success_count += 1

                    # Save thumbnails if output directory specified
                    if output_dir:
                        file_output_dir = Path(output_dir) / file_path.stem
                        thumbnail_paths = views_to_thumbnails(
                            extracted_views, output_dir=file_output_dir
                        )

                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    click.echo(f"✗ {file_path.name}: {e}", err=True)
                    error_count += 1

            # Summary
            click.echo(
                f"\nCompleted: {success_count} files processed, {total_views} views extracted, {error_count} failed"
            )

            if error_count > 0:
                sys.exit(1)
        else:
            click.echo(
                f"Error: {input_path} is neither a file nor a directory", err=True
            )
            sys.exit(1)
    else:
        # Neither OMERO nor file input provided
        click.echo(
            "Error: Please provide either a local file/directory path or "
            "--host and --iid for OMERO access",
            err=True,
        )
        sys.exit(1)


@cli.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--source",
    "-s",
    type=click.Choice(["auto", "bioio", "omero", "zarr"], case_sensitive=False),
    default="auto",
    help="Data source type (default: auto-detect)",
)
@click.option(
    "--host",
    help="OMERO server hostname (e.g., omero.iscc.id)",
)
@click.option(
    "--iid",
    type=int,
    help="OMERO image ID",
)
def pixhash(input, source, host, iid):
    """
    Generate normalized pixel hashes for bioimage data.

    Produces reproducible SHA1 hashes over normalized pixel data from various sources:
    - Local bioimage files (using BioIO)
    - OMERO server images (using BlitzGateway)
    - OME-Zarr files

    All implementations produce identical hashes for the same image data.
    """
    hashes = []

    # OMERO mode
    if host and iid:
        try:
            click.echo(f"Connecting to OMERO server: {host}")
            hashes = pixhash_omero(host, iid)
            click.echo(f"✓ Generated {len(hashes)} hash(es) from OMERO:")
            for i, h in enumerate(hashes):
                click.echo(f"  Image {i}: {h}")
        except Exception as e:
            click.echo(f"✗ Error with OMERO: {e}", err=True)
            sys.exit(1)

    # Local file mode
    elif input:
        input_path = Path(input)

        if not input_path.exists():
            click.echo(f"✗ File not found: {input_path}", err=True)
            sys.exit(1)

        # Auto-detect source type
        if source == "auto":
            if input_path.suffix == ".zarr" or (
                input_path.is_dir() and (input_path / ".zattrs").exists()
            ):
                source = "zarr"
            else:
                source = "bioio"
            logger.debug(f"Auto-detected source type: {source}")

        try:
            if source == "zarr":
                click.echo(f"Processing OME-Zarr: {input_path}")
                hashes = pixhash_zarr(str(input_path))
                click.echo(f"✓ Generated {len(hashes)} hash(es):")
                for i, h in enumerate(hashes):
                    click.echo(f"  Series {i}: {h}")
            else:  # bioio
                click.echo(f"Processing bioimage: {input_path}")
                hashes = pixhash_bioio(str(input_path))
                click.echo(f"✓ Generated {len(hashes)} hash(es):")
                for i, h in enumerate(hashes):
                    click.echo(f"  Scene {i}: {h}")
        except Exception as e:
            click.echo(f"✗ Error processing {input_path}: {e}", err=True)
            sys.exit(1)

    else:
        click.echo(
            "Error: Please provide either a local file path or "
            "--host and --iid for OMERO access",
            err=True,
        )
        sys.exit(1)

    return hashes


@cli.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save view PNGs (optional)",
)
@click.option(
    "--max-views",
    "-n",
    default=5,
    type=int,
    help="Maximum views per scene (default: 5)",
)
def biocode(input, output_dir, max_views):
    """
    Generate bioimage fingerprints with ISCC-SUM and ISCC-MIXED codes.

    Creates comprehensive bioimage fingerprints by:
    - Generating ISCC-SUM hash over normalized pixel content
    - Extracting ~5 representative views per scene
    - Generating ISCC-IMAGE codes for each view
    - Combining view codes into a global ISCC-MIXED descriptor

    Saves selected views as PNG files if --output-dir is specified.
    """
    input_path = Path(input)

    if not input_path.is_file():
        click.echo(f"Error: {input_path} is not a file", err=True)
        sys.exit(1)

    try:
        logger.info(f"Generating biocode for: {input_path}")

        # Generate fingerprints
        fingerprints = generate_biocode(
            str(input_path), output_dir=output_dir, max_views=max_views
        )

        # Format and display output
        output = format_output(fingerprints, input_path.name)
        click.echo(output)

        # Summary
        total_views = sum(len(fp.views) for fp in fingerprints)
        click.echo(
            f"✓ Generated {len(fingerprints)} scene fingerprint(s) with {total_views} total views"
        )

        if output_dir:
            click.echo(f"✓ Saved views to: {output_dir}")

    except Exception as e:
        click.echo(f"✗ Error processing {input_path}: {e}", err=True)
        logger.exception("Biocode generation failed")
        sys.exit(1)


if __name__ == "__main__":
    cli()
