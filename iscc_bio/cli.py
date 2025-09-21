"""CLI interface for iscc-bio."""

import click
from pathlib import Path
import sys
import logging
from iscc_bio.extract import extract_thumbnail


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


if __name__ == "__main__":
    cli()
