# iscc-bio - ISCC Processing for Bioimage Data

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/bio-codes/iscc-bio)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bio-codes/iscc-bio/blob/main/LICENSE)

**ISCC Processing for Multi-Dimensional Bioimage Data**

Generate [ISO 24138:2024](https://www.iso.org/standard/77899.html) International Standard Content Codes (ISCC)
for bioimage data across multiple formats using deterministic **IMAGEWALK** plane traversal.

## Project Status

**Version 0.1.0** - Unreleased.

!!! warning

    This package is a proof of concept under active development, and breaking changes may be released at any time.

## Overview

`iscc-bio` bridges bioimage formats with ISCC-CODE processing by implementing the **IMAGEWALK** specification -
a deterministic algorithm for traversing and canonicalizing pixel data from multi-dimensional bioimaging data.
This produces consistent, reproducible content identifiers regardless of source format or storage platform.

Documentation: https://bio.iscc.codes

### Key Features

- **Format-Agnostic Hashing**: Generate reproducible ISCCs at the level of pixel data across OME-TIFF, OME-Zarr,
    OMERO, CZI, ND2, LIF, and other formats
- **IMAGEWALK Implementation**: Deterministic Z→C→T plane traversal with canonical byte representation
- **Multi-Source Support**: Process local files (via BioIO), OME-Zarr archives, and OMERO remote servers
- **Memory Efficient**: Lazy loading with Dask for processing large multi-dimensional images
- **Multi-Scene Processing**: Handle complex multi-scene/multi-series bioimage files
- **Command-Line Tools**: CLI commands for code generation, pixel hashing, and view extraction

## Installation

### Basic Installation

```bash
# Using uv (recommended)
uv tool install iscc-bio

# Using pip
pip install iscc-bio
```

### Installation with Format Support

```bash
# Install with all bioimage reader plugins
uv tool install "iscc-bio[readers]"

# Install with specific format support
uv tool install "iscc-bio[czi,nd2,lif]"

# Install with OMERO support
uv tool install "iscc-bio[omero]"

# Install everything
uv tool install "iscc-bio[all]"
```

### Available Optional Dependencies

- **readers**: All BioIO reader plugins (BioFormats, CZI, OME-TIFF, OME-Zarr, ND2, LIF, etc.)
- **omero**: OMERO Blitz gateway for remote server access
- **bioformats**: BioFormats reader for broad format support
- **czi**, **nd2**, **lif**, **ome-tiff**, **ome-zarr-plugin**, **dv**, **tifffile**: Individual format readers

## Quick Start

### Eperimantal CLI scripts

#### Generate Bioimage Fingerprint

```bash
# Generate ISCC-based bioimage fingerprint
iscc-bio biocode myimage.czi

# Output includes:
# - ISCC-SUM hash over normalized pixel content
# - Representative view extraction (~5 views per scene)
# - ISCC-IMAGE codes for each view
# - ISCC-MIXED global descriptor
```

#### Pixel Hash (IMAGEWALK)

```bash
# Generate reproducible pixel hash using IMAGEWALK
iscc-bio pixhash myimage.ome.tiff

# Works with multiple sources:
iscc-bio pixhash local/file.czi           # Local bioimage file
iscc-bio pixhash data.zarr                # OME-Zarr/NGFF
iscc-bio pixhash --host omero.server.com --iid 123  # OMERO server
```

#### Extract Representative Views

```bash
# Extract intelligent 2D views for perceptual hashing
iscc-bio views myimage.nd2 --output-dir ./views/

# Extraction strategies:
# - Maximum intensity projections (MIP)
# - Best focus planes
# - Representative sampling
# - Multi-channel composites
```

## IMAGEWALK Specification

**IMAGEWALK** is a deterministic algorithm for traversing multi-dimensional bioimage data to produce
format-agnostic, reproducible hash digests.

### Core Principles

1. **Z→C→T Traversal Order**: Planes are processed in deterministic order:

    - Outermost loop: Z dimension (depth/focal plane)
    - Middle loop: C dimension (channel)
    - Innermost loop: T dimension (time)

2. **Canonical Byte Representation**: Each 2D plane is:

    - Flattened in row-major order (Y then X)
    - Encoded as big-endian bytes
    - Fed to a hash processor

3. **Multi-Scene Independence**: Each scene/series is processed separately, producing one hash per scene

### Example Traversal

For an image with `Z=2, C=3, T=2` dimensions (12 total planes):

```
Plane 1:  z=0, c=0, t=0    Plane 7:  z=1, c=0, t=0
Plane 2:  z=0, c=0, t=1    Plane 8:  z=1, c=0, t=1
Plane 3:  z=0, c=1, t=0    Plane 9:  z=1, c=1, t=0
Plane 4:  z=0, c=1, t=1    Plane 10: z=1, c=1, t=1
Plane 5:  z=0, c=2, t=0    Plane 11: z=1, c=2, t=0
Plane 6:  z=0, c=2, t=1    Plane 12: z=1, c=2, t=1
```

### Implementation Modules

- **`iw_bioio.py`**: BioIO-based implementation for local files
- **`iw_ngff.py`**: OME-NGFF/Zarr implementation using ome-zarr-py
- **`iw_blitz.py`**: OMERO Blitz implementation for remote servers

All implementations produce identical hashes for identical pixel data, conforming to the
[IMAGEWALK specification](https://github.com/bio-codes/iscc-bio/blob/main/docs/imagewalk.md).

## Command-Line Interface

### `biocode` - Generate Bioimage Fingerprint

Create comprehensive bioimage fingerprints with ISCC codes:

```bash
iscc-bio biocode INPUT [OPTIONS]

Options:
  -o, --output-dir PATH    Save extracted view PNGs
  -n, --max-views INTEGER  Maximum views per scene (default: 5)
```

### `pixhash` - Normalized Pixel Hash

Generate reproducible SHA1 hashes over normalized pixel data:

```bash
iscc-bio pixhash INPUT [OPTIONS]

Options:
  -s, --source [auto|bioio|omero|zarr]  Data source type
  --host TEXT                           OMERO server hostname
  --iid INTEGER                         OMERO image ID
```

### `views` - Extract Representative Views

Extract intelligent 2D views for perceptual hashing:

```bash
iscc-bio views INPUT [OPTIONS]

Options:
  -s, --strategies TEXT    View strategies (mip, best_focus, representative, composite)
  -n, --max-views INTEGER  Maximum views to extract (default: 8)
  -o, --output-dir PATH    Directory to save thumbnails
  --host TEXT              OMERO server hostname
  --iid INTEGER            OMERO image ID
```

### `scenes` - Extract Scene Thumbnails

Extract thumbnails from all scenes in a multi-scene file:

```bash
iscc-bio scenes INPUT
```

### `thumb` - Extract Thumbnail

Extract a single representative thumbnail from a bioimage:

```bash
iscc-bio thumb INPUT
```

## Python API

### IMAGEWALK Plane Iteration

```python
from iscc_bio.imagewalk.iw_bioio import iter_planes_bioio
from iscc_bio.imagewalk.iw_ngff import iter_planes_ngff
from iscc_bio.imagewalk.iw_blitz import iter_planes_blitz

# Iterate over planes using BioIO
for plane in iter_planes_bioio("image.czi"):
    print(f"Scene {plane.scene_idx}, Z={plane.z_depth}, "
          f"C={plane.c_channel}, T={plane.t_time}")
    print(f"Shape: {plane.xy_array.shape}, dtype: {plane.xy_array.dtype}")

# Iterate over OME-Zarr planes
for plane in iter_planes_ngff("data.zarr"):
    # Process plane.xy_array (2D numpy array)
    pass

# Iterate over OMERO planes
from omero.gateway import BlitzGateway
conn = BlitzGateway("user", "pass", host="omero.server.com")
conn.connect()
image = conn.getObject("Image", 123)

for plane in iter_planes_blitz(image):
    # Process plane.xy_array
    pass
conn.close()
```

### Generate Biocode

```python
from iscc_bio.biocode import generate_biocode, format_output

# Generate bioimage fingerprints
fingerprints = generate_biocode("image.nd2", max_views=5)

# Format output
output = format_output(fingerprints, "image.nd2")
print(output)
```

### Pixel Hashing

```python
from iscc_bio.pixhash import pixhash_bioio, pixhash_zarr, pixhash_omero

# Generate pixel hash (returns list of hashes, one per scene)
hashes = pixhash_bioio("image.lif")
print(hashes[0])  # Hash for first scene

# OME-Zarr
hashes = pixhash_zarr("data.zarr")

# OMERO
hashes = pixhash_omero("omero.server.com", image_id=123)
```

## Supported Formats

Via BioIO plugin ecosystem:

- **OME-TIFF/TIFF**: Multi-page TIFF with OME-XML metadata
- **OME-Zarr/NGFF**: Next-generation file format
- **OMERO**: Remote server access via Blitz gateway
- **CZI**: Carl Zeiss Image format
- **ND2**: Nikon NIS-Elements format
- **LIF**: Leica Image File format
- **DV**: DeltaVision format
- **BioFormats**: 150+ formats via Bio-Formats Java library

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/bio-codes/iscc-bio.git
cd iscc-bio

# Install with all dependencies
uv sync --extra all

# Run CLI during development
uv run iscc-bio --help
```

### Development Commands

This project uses [poethepoet](https://github.com/nat-n/poethepoet) for task automation:

```bash
# Format markdown files
uv run poe format-md

# Format code files
uv run poe format-code

# Build documentation
uv run poe docs-build

# Run all formatting and docs
uv run poe all
```

## Architecture

### Core Modules

- **`iscc_bio.imagewalk`**: IMAGEWALK plane traversal implementations

    - `iw_bioio.py`: BioIO implementation
    - `iw_ngff.py`: OME-Zarr/NGFF implementation
    - `iw_blitz.py`: OMERO Blitz implementation
    - `models.py`: Plane data model

- **`iscc_bio.biocode`**: ISCC bioimage fingerprint generation

- **`iscc_bio.pixhash`**: Normalized pixel hashing across sources

- **`iscc_bio.views`**: Intelligent view extraction strategies

- **`iscc_bio.cli`**: Command-line interface

### Design Principles

1. **Lazy Loading**: Uses Dask arrays for memory-efficient processing of large images
2. **Format Agnostic**: Identical processing logic across all formats via IMAGEWALK
3. **Deterministic**: Reproducible hashes across platforms and formats
4. **Modular**: Clean separation between traversal, canonicalization, and hashing

## Funding

This work was supported through the Open Science Clusters’ Action for Research and Society (OSCARS) European
project under grant agreement Nº101129751.

See:
[BIO-CODES](https://oscars-project.eu/projects/bio-codes-enhancing-ai-readiness-bioimaging-data-content-based-identifiers)
project (Enhancing AI-Readiness of Bioimaging Data with Content-Based Identifiers).

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Citation

If you use iscc-bio in your research, please cite:

```bibtex
@software{iscc_bio,
  title        = {bio-codes/iscc-bio: ISCC Processing for Bioimage Data},
  author       = {Pan, Titusz},
  year         = 2025,
  url          = {https://github.com/bio-codes/iscc-bio},
  note         = {Supported by OSCARS (Open Science Clusters' Action for Research and Society) under European Commission grant agreement Nº101129751},
  version      = {0.1.0}
}
```

## Related Projects

- [iscc-sum](https://github.com/bio-codes/iscc-sum) - Fast ISCC Data-Code and Instance-Code hashing
- [iscc-core](https://github.com/iscc/iscc-core) - ISCC Core Algorithms
- [BioIO](https://github.com/bioio-devs/bioio) - Bioimage reading library
- [OME-Zarr](https://github.com/ome/ome-zarr-py) - Next-generation file format implementation
