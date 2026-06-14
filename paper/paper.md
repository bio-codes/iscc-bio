---
title: 'iscc-bio: Deterministic content identifiers for bioimage data'
tags:
  - Python
  - bioimaging
  - microscopy
  - content identification
  - reproducibility
  - OME-TIFF
  - OME-Zarr
authors:
  - name: Titusz Pan
    orcid: 0000-0002-0521-4214
    affiliation: 1
affiliations:
  - name: ISCC Foundation, Germany
    index: 1
date: 14 June 2026
bibliography: paper.bib
---

# Summary

`iscc-bio` is a Python package for generating International Standard Content Code (ISCC) identifiers [@iso24138]
for multi-dimensional bioimage data. Bioimages are commonly stored as OME-TIFF, OME-NGFF/Zarr, OMERO-managed
pixel stores, and vendor-specific microscopy formats. These containers differ in metadata layout, chunking,
compression, tiling, and scene representation, even when they describe the same pixel content. `iscc-bio`
addresses this by applying a deterministic IMAGEWALK traversal over image planes and feeding a canonical byte
representation into ISCC Data-Code and Instance-Code hashers.

The package exposes both a Python API and command-line interface for local BioIO-readable files, OME-NGFF/Zarr
stores, and OMERO sources. It is intended as a small interoperability layer between the microscopy software
ecosystem and content-identification infrastructure: the input may be a bioimage container, but the identifier
is derived from normalized pixel planes in deterministic scene, Z, channel, and time order.

# Statement of need

Scientific bioimage collections increasingly combine proprietary microscope formats, OME-TIFF exports,
cloud-native OME-NGFF/Zarr stores, and repository systems such as OMERO [@ome; @ome_ngff; @omero]. Cryptographic
file checksums are valuable for bitstream integrity, but they are not enough for cross-format bioimage
workflows: a TIFF-to-Zarr conversion, a different chunk shape, or a metadata-preserving reserialization changes
the file bytes even when the decoded pixel planes are unchanged. Researchers, archives, and infrastructure
projects therefore need reproducible identifiers that can be computed after decoding image content, while still
preserving enough determinism for independent verification.

`iscc-bio` provides this missing pixel-canonical layer for the ISCC ecosystem. It uses BioIO reader plugins
[@bioio] and OME-Zarr tooling [@ome_zarr_py] to traverse bioimage planes without loading an entire dataset into
memory. For each scene, IMAGEWALK visits planes in deterministic `Z -> C -> T` order. Every plane is converted
to a canonical row-major byte representation before hash updates. The result is an ISCC composite code with data
and instance units over normalized bioimage content rather than the original container byte stream.

This scope is deliberately narrower than semantic image understanding. The experiment included with the
repository tests whether conversions preserve the same pixel-canonical BioCode. It does not claim that raw files
have the same cryptographic checksum, nor that perceptually similar but pixel-different images should match.
Lossy compression, intensity rescaling, channel projection, or non-canonical rendering choices can correctly
produce different identifiers.

# State of the field

Bioimage data-management tools such as OME-TIFF, OME-NGFF/Zarr, BioIO, and OMERO make microscopy data readable
across laboratories and archives [@ome; @ome_ngff; @bioio; @omero]. They do not, by themselves, define a
container-independent content identifier for decoded bioimage pixels. Conversely, generic cryptographic hashes
identify exact byte streams but intentionally change after benign reserialization. `iscc-bio` contributes a
focused bridge between these layers: it uses existing microscopy readers and storage models, but derives ISCC
BioCodes from a deterministic traversal of decoded pixel planes.

# Software design

The package separates source access, traversal, and code generation. Source-specific iterators produce a common
`Plane` structure for BioIO-readable files, OME-NGFF/Zarr stores, and OMERO sources. `generate_biocode` consumes
that iterator without needing to know the original container, updates ISCC data and instance hashers with
canonical row-major plane bytes, and emits one result per scene. This keeps the matching claim local and
auditable: if two sources yield the same ordered canonical planes, they should yield the same scene-level
BioCode.

# Research impact

Stable content-derived identifiers are useful for deduplicating repository holdings, checking whether conversion
pipelines preserve pixel content, linking files and derived OME-NGFF stores, and indexing per-plane simprints for
more granular search. The included JOSS experiment is deliberately modest: it verifies `iscc-bio` round-trip
behavior across OME-TIFF and OME-Zarr encodings for small public samples and synthetic fixtures. Larger accuracy,
lossy-compression, and independent-converter benchmarks remain future work and are better suited to a dedicated
benchmarking repository.

# Functionality

The core package provides:

- deterministic IMAGEWALK plane traversal for BioIO-readable files, OME-NGFF/Zarr stores, and OMERO images;
- scene-level BioCode generation using ISCC Data-Code and Instance-Code units over canonical plane bytes;
- optional per-plane simprints for more granular similarity indexing;
- representative-view extraction and experimental image-code generation for thumbnails or selected 2D views;
- a command-line interface for `biocode`, `imagecode`, `views`, `scenes`, and thumbnail generation.

The design follows the OME model's separation between image structure and storage container [@ome]. Format
support is delegated to the BioIO plugin ecosystem so that project code can focus on deterministic traversal and
canonicalization.

# Reproducible conversion experiment

The repository includes a bounded experiment in `experiments/joss_conversion_matching.py`. It acquires a small
public corpus from Open Microscopy sample data, including OME-TIFF, TIFF, Olympus OIR, and Leica LIF examples
when the corresponding readers are installed. The manifest pins both byte sizes and SHA-256 digests for the
public downloads. For each readable sample, the script:

1. downloads the sample into an ignored local cache, refusing files above a fixed size limit;
2. computes the original scene-level BioCode with `iscc-bio`;
3. materializes each scene from `iscc-bio`'s own IMAGEWALK plane extraction as a dense `TCZYX` array;
4. writes normalized OME-TIFF and single-scale OME-Zarr round-trip variants;
5. recomputes BioCodes for the converted variants; and
6. records whether each scene's composite code and component units match.

The experiment is intentionally small enough to run during review while still covering multiple public
microscopy sources. It reports JSON and CSV artifacts under `experiments/results/`. Network access is not
required for tests: the test suite creates synthetic multi-Z, multi-channel data and verifies that OME-TIFF and
OME-Zarr round trips preserve `iscc-bio` BioCodes.

The experiment also makes failures explicit. If an optional BioIO reader is not installed or cannot decode a
public source, the result row records that error rather than silently removing the sample. This is important for
reproducibility in bioimaging, where format support often depends on optional plugins, Java/Bio-Formats
integration, or reader version details.

# Related work

The International Standard Content Code (ISCC) specifies content-derived identifiers for digital media
[@iso24138]. The `iscc-bio` package builds on this identifier model for scientific image data. It relies on
BioIO [@bioio] for reader abstraction, OME-TIFF and OME-NGFF/Zarr for interoperable microscopy data models
[@ome; @ome_ngff], and OMERO for repository-backed image management [@omero]. Compared with general file
hashing, `iscc-bio` computes over decoded, canonicalized image planes. Compared with semantic or perceptual
image matching, it makes a stricter reproducibility claim: matching identifiers are evidence that the compared
inputs produced matching canonical pixel streams under the IMAGEWALK procedure.

# AI usage disclosure

Large language model coding assistants (OpenAI Codex CLI and Anthropic Claude Code) were used during preparation
of this paper draft and the accompanying experiment implementation. The repository changes were checked with
local tests before submission.

# Acknowledgements

This work is developed in the BioCodes and ISCC ecosystem. Public sample images used by the reproducibility
experiment are provided through Open Microscopy sample image repositories and related community datasets.

# References
