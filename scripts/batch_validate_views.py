"""
Batch process all bioimages in a directory with the validation script.
"""

import subprocess
import sys
from pathlib import Path


def process_all_bioimages(directory: Path):
    """Process all bioimage files in the directory."""

    # Common bioimage extensions
    extensions = ['.czi', '.nd2', '.tif', '.tiff', '.lif', '.oir', '.ims']

    # Find all bioimage files
    bioimage_files = []
    for ext in extensions:
        bioimage_files.extend(directory.glob(f'*{ext}'))

    print(f"Found {len(bioimage_files)} bioimage files in {directory}")
    print("=" * 60)

    success_count = 0
    failed_files = []

    for i, filepath in enumerate(bioimage_files, 1):
        print(f"\n[{i}/{len(bioimage_files)}] Processing: {filepath.name}")
        print("-" * 40)

        # Generate output filename
        output_file = filepath.parent / f"{filepath.stem}-views.png"

        # Run the validation script
        cmd = [
            sys.executable,
            "scripts/validate_bioimage_views.py",
            str(filepath),
            "--strategy", "comprehensive",
            "--save", str(output_file),
            "--no-display"  # Don't show interactive plots
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per file
            )

            if result.returncode == 0:
                print(f"✓ Success! Views saved to: {output_file.name}")
                # Print extraction summary if available
                if "EXTRACTION SUMMARY" in result.stdout:
                    lines = result.stdout.split('\n')
                    for j, line in enumerate(lines):
                        if "EXTRACTION SUMMARY" in line:
                            # Print summary section
                            for k in range(j-1, min(j+10, len(lines))):
                                if lines[k].strip():
                                    print(f"  {lines[k]}")
                success_count += 1
            else:
                print(f"✗ Failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                failed_files.append(filepath.name)

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout: Processing took longer than 2 minutes")
            failed_files.append(filepath.name)
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_files.append(filepath.name)

    # Final summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {success_count}/{len(bioimage_files)} files")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")

    print(f"\nOutput files saved in: {directory}")


if __name__ == "__main__":
    bioimage_dir = Path("E:/biocodes/bioimages")

    if not bioimage_dir.exists():
        print(f"Error: Directory not found: {bioimage_dir}")
        sys.exit(1)

    process_all_bioimages(bioimage_dir)