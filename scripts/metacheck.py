"""Check common metadata fields across bioimage files in a folder.

Iterates over all files in a folder and attempts to read metadata using bioio
without loading image data into memory. Reports which metadata properties are
available and have values set across ALL files.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import iscc_bio  # noqa: F401 - Import for bioformats initialization
from bioio import BioImage
from loguru import logger


# Metadata properties to check (excluding methods and data properties)
METADATA_PROPS = [
    "channel_names",
    "current_scene",
    "current_resolution_level",
    "dims",
    "physical_pixel_sizes",
    "resolution_levels",
    "scale",
    "scenes",
    "shape",
    "time_interval",
]

# Structured metadata objects to check
STRUCTURED_METADATA = [
    "standard_metadata",
    "ome_metadata",
]


def is_scalar(value: Any) -> bool:
    """Check if value is scalar (not a complex object or collection).

    Args:
        value: Value to check

    Returns:
        True if value is scalar
    """
    return isinstance(value, (str, int, float, bool, type(None)))


def extract_object_fields(
    obj: Any, prefix: str = "", max_depth: int = 5, current_depth: int = 0
) -> Dict[str, Any]:
    """Recursively extract scalar fields from an object.

    Args:
        obj: Object to extract fields from
        prefix: Prefix for nested field names
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth

    Returns:
        Dictionary of field names to values (only scalars)
    """
    fields = {}

    if obj is None or current_depth >= max_depth:
        return fields

    # Handle lists/tuples - extract from first few items
    if isinstance(obj, (list, tuple)):
        # Only process first 3 items to avoid too much data
        for idx, item in enumerate(obj[:3]):
            if is_scalar(item):
                field_name = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                fields[field_name] = item
            elif hasattr(item, "__dict__") or isinstance(item, dict):
                nested = extract_object_fields(
                    item, f"{prefix}[{idx}]", max_depth, current_depth + 1
                )
                fields.update(nested)
        return fields

    # Handle dataclass-like objects
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if key.startswith("_"):
                continue

            field_name = f"{prefix}.{key}" if prefix else key

            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                continue

            # Only include scalar values directly
            if is_scalar(value):
                fields[field_name] = value
            # Recurse into nested objects, dicts, or lists
            elif hasattr(value, "__dict__") or isinstance(value, (dict, list, tuple)):
                nested = extract_object_fields(
                    value, field_name, max_depth, current_depth + 1
                )
                fields.update(nested)

    # Handle dict-like objects
    elif isinstance(obj, dict):
        for key, value in obj.items():
            field_name = f"{prefix}.{key}" if prefix else str(key)

            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                continue

            if is_scalar(value):
                fields[field_name] = value
            elif isinstance(value, (dict, list, tuple)) or hasattr(value, "__dict__"):
                nested = extract_object_fields(
                    value, field_name, max_depth, current_depth + 1
                )
                fields.update(nested)

    return fields


def get_metadata_dict(img: BioImage) -> Dict[str, Any]:
    """Extract metadata properties from BioImage without loading image data.

    Args:
        img: BioImage object to extract metadata from

    Returns:
        Dictionary of metadata property names to values
    """
    metadata = {}

    # Get simple properties
    for prop_name in METADATA_PROPS:
        try:
            value = getattr(img, prop_name)
            # Only include if value is not None and not empty
            if value is not None:
                # Check for empty collections
                if hasattr(value, "__len__") and len(value) == 0:
                    continue
                metadata[prop_name] = value
        except Exception as e:
            logger.debug(f"Failed to get {prop_name}: {e}")

    # Get structured metadata fields
    for struct_name in STRUCTURED_METADATA:
        try:
            struct_obj = getattr(img, struct_name)
            fields = extract_object_fields(struct_obj, prefix=struct_name)
            metadata.update(fields)

            # Special handling for OME instruments
            if struct_name == "ome_metadata" and hasattr(struct_obj, "instruments"):
                instruments = struct_obj.instruments
                if instruments:
                    for idx, instrument in enumerate(instruments):
                        inst_fields = extract_object_fields(
                            instrument, f"ome_metadata.instrument_{idx}"
                        )
                        metadata.update(inst_fields)

        except Exception as e:
            logger.debug(f"Failed to get {struct_name}: {e}")

    return metadata


def check_folder_metadata(folder_path: Path) -> None:
    """Check common metadata across all bioimage files in a folder.

    Args:
        folder_path: Path to folder containing bioimage files
    """
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        sys.exit(1)

    if not folder_path.is_dir():
        logger.error(f"Not a directory: {folder_path}")
        sys.exit(1)

    # Collect metadata from all files (only in specified folder, not subfolders)
    files_metadata: List[tuple[Path, Dict[str, Any]]] = []
    all_files = [f for f in folder_path.iterdir() if f.is_file()]

    logger.info(f"Scanning {len(all_files)} files in {folder_path}")

    for file_path in all_files:
        try:
            logger.debug(f"Reading {file_path.name}")
            img = BioImage(file_path)
            metadata = get_metadata_dict(img)

            if metadata:
                files_metadata.append((file_path, metadata))
                logger.debug(f"✓ {file_path.name} - {len(metadata)} properties")
            else:
                logger.debug(f"✗ {file_path.name} - no metadata")

        except Exception as e:
            logger.debug(f"✗ {file_path.name} - {type(e).__name__}: {e}")
            continue

    if not files_metadata:
        logger.warning("No bioimage files with metadata found")
        return

    logger.info(f"Successfully read metadata from {len(files_metadata)} files")

    # Find common properties across all files
    common_props: Set[str] | None = None

    for file_path, metadata in files_metadata:
        file_props = set(metadata.keys())
        if common_props is None:
            common_props = file_props
        else:
            common_props &= file_props

    if not common_props:
        logger.warning("No common metadata properties found across all files")
        return

    # Sort for consistent output
    common_props_sorted = sorted(common_props)

    logger.info(f"\nCommon metadata properties ({len(common_props_sorted)}):")
    print("\nMetadata properties available on ALL files with values set:")
    print("=" * 60)

    for prop in common_props_sorted:
        print(f"  • {prop}")

        # Show example values from first 3 files
        example_values = []
        for file_path, metadata in files_metadata[:3]:
            value = metadata[prop]
            # Format value for display
            if hasattr(value, "__dict__"):
                # Object with attributes
                value_str = f"{type(value).__name__}(...)"
            elif isinstance(value, (tuple, list)) and len(value) > 5:
                value_str = f"{type(value).__name__}[{len(value)} items]"
            else:
                value_str = str(value)
            example_values.append(f"{file_path.name}: {value_str}")

        for example in example_values:
            print(f"      {example}")
        print()

    print("=" * 60)
    print(f"Total: {len(common_props_sorted)} common properties")

    # Find interesting vendor/instrument metadata that may not be in all files
    all_props: Set[str] = set()
    for _, metadata in files_metadata:
        all_props.update(metadata.keys())

    # Filter for vendor/microscope-related properties
    vendor_keywords = [
        "vendor",
        "manufacturer",
        "model",
        "microscope",
        "instrument",
        "objective",
    ]
    vendor_props = {
        prop
        for prop in all_props
        if any(keyword in prop.lower() for keyword in vendor_keywords)
    }

    if vendor_props:
        print("\n" + "=" * 60)
        print("\nVendor/Instrument metadata (may not be in all files):")
        print("=" * 60)

        for prop in sorted(vendor_props):
            # Count how many files have this property
            files_with_prop = [
                (fp, md[prop]) for fp, md in files_metadata if prop in md
            ]

            print(f"  • {prop} ({len(files_with_prop)}/{len(files_metadata)} files)")

            # Show example values
            for file_path, value in files_with_prop[:3]:
                value_str = str(value)[:80]  # Truncate long values
                print(f"      {file_path.name}: {value_str}")
            print()


def main():
    """Main entry point."""
    # Default test folder
    default_folder = Path("E:/biocodes/bioimages")

    if len(sys.argv) > 1:
        folder_path = Path(sys.argv[1])
    else:
        folder_path = default_folder

    logger.info(f"Checking metadata in: {folder_path}")
    check_folder_metadata(folder_path)


if __name__ == "__main__":
    main()
