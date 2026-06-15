---
title: 'iscc-bio: Deterministic content identifiers for bioimage data'
author: Titusz Pan
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
    affiliation: "1"
    corresponding: true
    email: tp@iscc.io
affiliations:
  - name: ISCC Foundation, Germany
    index: 1
date: 14 June 2026
bibliography: paper.bib
---

# Summary

`iscc-bio` is a Python package for generating scene-level, content-derived identifiers for multi-dimensional
bioimage data using the International Standard Content Code (ISCC) framework [@iso24138; @iscc_core]. Bioimages
are commonly stored as OME-TIFF, OME-NGFF/Zarr, OMERO-managed pixel stores, and vendor-specific microscopy
formats. These containers differ in metadata layout, chunking, compression, tiling, and scene representation,
even when they describe the same pixel content. `iscc-bio` addresses this by applying a project-defined
IMAGEWALK traversal over image planes and feeding canonical bytes into ISCC-SUM Data-Code and Instance-Code
hashers. In the experiment, the Instance-Code checks strict equality over the canonical stream, while the
Data-Code is compared by Hamming distance for small decoded-data perturbations.

The package exposes both a Python API and command-line interface for local BioIO-readable files, OME-NGFF/Zarr
stores, and OMERO sources. It is intended as a small interoperability layer between the microscopy software
ecosystem and content-identification infrastructure: the input may be a bioimage container, but the identifier
is derived from normalized pixel planes in deterministic scene, Z, channel, and time order.

# Statement of need

Scientific bioimage collections combine proprietary microscope formats, OME-TIFF exports, OME-NGFF/Zarr stores,
and repository systems such as OMERO [@ome; @ome_ngff; @omero]. File checksums verify bitstream integrity, but a
TIFF-to-Zarr conversion, different chunk shape, or metadata-preserving reserialization changes bytes even when
decoded pixels are unchanged. Researchers and archives therefore need identifiers computed after decoding image
content while preserving deterministic verification.

`iscc-bio` provides a pixel-canonical layer for the ISCC ecosystem. It uses BioIO reader plugins
[@bioio] and OME-Zarr tooling [@ome_zarr_py] to traverse bioimage planes without loading an entire dataset into
memory. For each scene, IMAGEWALK visits planes in deterministic `Z -> C -> T` order. Every plane is converted
to a canonical row-major byte representation before hash updates. The result is a BioCode: a scene-level
ISCC-CODE composed of Data-Code and Instance-Code units over normalized bioimage content rather than the original
container byte stream. The paper experiment uses the 256-bit WIDE ISCC-CODE subtype implemented by `iscc-core`,
which is an extension to ISO 24138:2024 rather than the ISO-conformant narrow encoding.

This scope is deliberately narrower than semantic image understanding. Exact `iscc-bio` BioCode equality means
that compared inputs produced the same canonical pixel stream under IMAGEWALK. When a conversion introduces a
small decoded-pixel perturbation, exact equality should fail: the Instance-Code is a BLAKE3-based digest and
changes [@iscc_core]. For the controlled one-pixel drifts tested here, the Data-Code remained within a low
Hamming-distance search threshold. The optional Content-Code-Mixed sidecar samples up to three IMAGEWALK planes,
computes standard ISCC Image Content-Codes, and combines them with the standard mixed-code construction; in the
tested conditions it supports content-layer comparison without changing Data-Code or Instance-Code semantics. Lossy
compression, intensity rescaling, channel projection, or non-canonical rendering choices can still produce large
distances, so the package reports component codes and distances rather than collapsing every comparison to a single
pass/fail value.

# State of the field

OME-TIFF, OME-NGFF/Zarr, BioIO, and OMERO make microscopy data readable across tools [@ome; @ome_ngff; @bioio;
@omero], but do not define a container-independent identifier for decoded bioimage pixels. Generic cryptographic
hashes identify exact byte streams. `iscc-bio` bridges these layers by deriving BioCodes from deterministic decoded
plane traversal.

# Software design

The package separates source access, traversal, and code generation. Source-specific iterators produce a common
`Plane` structure for BioIO-readable files, OME-NGFF/Zarr stores, and OMERO sources. `generate_biocode` consumes
that iterator, updates ISCC data and instance hashers with canonical row-major plane bytes, and emits one result
per scene. This keeps the matching claim local: if two sources yield the same ordered canonical planes, they should
yield the same scene-level BioCode. Matching across readers or converters depends on those tools exposing
equivalent plane order and pixel values, not merely on containers describing the same acquisition.

# Research impact statement

Stable content-derived identifiers can support deduplicating repository holdings, checking whether conversion
pipelines preserve pixel content, linking files and derived OME-NGFF stores, and indexing per-plane simprints for
more granular search. The included JOSS experiment is deliberately modest and directly exercises conversion
matching: it verifies OME-TIFF and OME-Zarr round trips for small public samples, and it injects controlled
one-pixel decoded-data drift to validate that Instance-Code equality fails while Data-Code search can still
recover near matches. Larger independent-converter and proprietary-codec benchmarks, including the known
CZI/JPEG-XR decoder-variance class, remain appropriate follow-up work once small unrestricted public fixtures are
available.

# Functionality

The core package provides:

- deterministic IMAGEWALK plane traversal for BioIO-readable files, OME-NGFF/Zarr stores, and OMERO images;
- scene-level BioCode generation using ISCC Data-Code and Instance-Code units over canonical plane bytes;
- optional Content-Code-Mixed sidecars from deterministic IMAGEWALK-selected planes for content-layer comparison;
- optional per-plane simprints for more granular similarity indexing;
- representative-view extraction and experimental image-code generation for thumbnails or selected 2D views;
- a command-line interface for `biocode`, `imagecode`, `views`, `scenes`, and thumbnail generation.

The design follows the OME model's separation between image structure and storage container [@ome]. Format
support is delegated to the BioIO plugin ecosystem so that project code can focus on deterministic traversal and
canonicalization.

# Reproducible conversion experiment

The repository includes a bounded experiment in `experiments/joss_conversion_matching.py`. It acquires a small
public corpus spanning OME-TIFF, plain TIFF, BioImage Archive TIFF, Zeiss CZI, Nikon ND2, Olympus OIR, and Leica
LIF samples. The manifest pins URLs, byte sizes, and SHA-256 digests for both public image downloads and external
converter archives. For each readable sample, the script downloads verified inputs into an ignored cache,
computes original scene-level BioCodes and optional Content-Code-Mixed sidecars, runs pinned Bio-Formats
`bfconvert` 8.5.0 [@bioformats], materializes IMAGEWALK planes as dense `TCZYX` arrays, writes OME-TIFF,
DEFLATE OME-TIFF, OME-Zarr, and deterministic one-pixel-drift variants, then records composite equality plus
Data-Code, Content-Code-Mixed, and Instance-Code Hamming distances.

The generated paper table is tracked in `paper/experiment-results.md`, and the script also writes the summary
figure. The current run used an illustrative 64-bit near-match threshold for 256-bit Data-Code units. Exact
`iscc-bio` OME-TIFF and OME-Zarr round trips matched for all six source samples that BioIO decoded locally.
Drifted conversions changed Instance-Code units in every case, while Data-Code distances stayed within threshold
and Content-Code-Mixed sidecars stayed identical for all tested drift rows. DEFLATE-compressed OME-TIFF round
trips also preserved the mixed sidecar for all decoded samples. Independent `bfconvert` conversions matched
exactly for OME-TIFF, plain TIFF, BioImage Archive TIFF, and ND2; CZI and LIF produced reader/converter-dependent
outputs, including LIF scene-count differences, rather than being hidden by the experiment. The Olympus
OIR fixture is retained in the pinned manifest but marked as skipped in this environment because the installed
reader stack could not decode it.

![Conversion matching outcomes. Exact rows preserve both Data-Code and Instance-Code. Near rows have changed Instance-Codes but Data-Code Hamming distance at or below the 64/256-bit threshold; Content-Code-Mixed equality is reported separately. \label{fig:conversion-matching}](figures/conversion-matching.png)

The generated CSV contains 40 comparison rows across seven public samples. Summarized by conversion condition:

- `bfconvert` to OME-TIFF produced nine scene comparisons: four exact matches for OME-TIFF, TIFF, BioImage
  Archive TIFF, and ND2, plus five mismatches for CZI/LIF reader-converter output and scene-count differences.
- `tifffile` OME-TIFF and DEFLATE-compressed OME-TIFF round trips produced six exact matches each.
- `ome-zarr-py` OME-Zarr round trips produced six exact matches.
- Deterministic one-pixel drift variants for OME-TIFF and OME-Zarr produced twelve near matches: all changed the
  Instance-Code, while Data-Code distances remained within the 64/256-bit threshold and Content-Code-Mixed units
  remained identical.
- The Olympus OIR source was retained in the pinned corpus and reported as a reader skip in this environment.

The full generated result table is kept in `paper/experiment-results.md` and the machine-readable CSV/JSON
artifacts are written under `experiments/results/`.

The default experiment now requires Java, downloads the pinned 51 MB Bio-Formats command-line archive, and runs
`bfconvert` as an actual independent converter. The larger `bioformats2raw` and `raw2ometiff` archives are pinned
in the manifest and can be enabled with `--external-tools all --allow-large-tool-downloads`, but they are gated
because their combined download size is over 400 MB. We also evaluated Bisque BioImage Convert (`imgcnv`)
[@bisque], which historically supported many biological image formats, but excluded it from the automated
experiment because the linked download/source infrastructure was unavailable during evaluation and no maintained
Python package or reliable pinned binary could be found.

The experiment is intentionally small enough to run during review while still covering multiple public microscopy
sources, target encodings, writer stacks, an independent Bio-Formats converter, and the Data-Code/Instance-Code
split that motivates ISCC-SUM in `iscc-bio`. It reports JSON and CSV artifacts under `experiments/results/`.
Network access is not required for tests: the test suite creates synthetic multi-Z, multi-channel data and verifies
that OME-TIFF and OME-Zarr round trips preserve `iscc-bio` BioCodes, that a fake `bfconvert` source-file converter
path is evaluated, and that a one-pixel perturbation verifies Data-Code near matching with Instance-Code mismatch.

The experiment also makes failures explicit. If an optional BioIO reader is not installed or cannot decode a
public source, the result row records a skip/error rather than silently removing the sample. The machine-readable
manifest records sample digests, resolved external tools, and relevant Python package versions, because format
support often depends on optional plugins, Java/Bio-Formats integration, or reader version details.

# Related work

The International Standard Content Code (ISCC) specifies content-derived identifiers for digital media
[@iso24138]. The `iscc-bio` package builds on this identifier model and the `iscc-core`/`iscc-lib` implementation
[@iscc_core] for scientific image data. It relies on
BioIO [@bioio] for reader abstraction, OME-TIFF and OME-NGFF/Zarr for interoperable microscopy data models
[@ome; @ome_ngff], and OMERO for repository-backed image management [@omero]. Compared with general file
hashing, `iscc-bio` computes over decoded, canonicalized image planes. Compared with semantic or perceptual
image matching, it makes a stricter reproducibility claim: matching identifiers are evidence that the compared
inputs produced matching canonical pixel streams under the IMAGEWALK procedure.

# AI usage disclosure

Large language model coding assistants were used during preparation of this paper draft and the accompanying
experiment implementation. The tools included OpenAI Codex CLI 0.139.0 and Anthropic Claude Code 2.1.175; they
were used for code-generation support, review, paper-edit suggestions, and verification planning. The human
author made the scientific and design decisions, selected the public datasets and result interpretation, reviewed
and edited generated text and code, and validated the repository changes with local tests and experiment runs
before submission.

# Acknowledgements

This work is developed in the BioCodes and ISCC ecosystem and was supported through the Open Science Clusters'
Action for Research and Society (OSCARS) European project under grant agreement Nº101129751. Public sample images
used by the reproducibility experiment are provided through Open Microscopy, BioImage Archive, and related
community sample repositories.

# References
