# Mixed Content-Code implementation note

## Question

Would an additional ISCC Content-Code-Mixed unit, generated from a deterministic subset of IMAGEWALK planes, improve `iscc-bio` robustness against conversions that add compression or tiny decoded-pixel drift?

## Architecture

The production implementation keeps the scene-level BioCode unchanged: it still contains the ISCC-SUM Data-Code and Instance-Code units over canonical IMAGEWALK bytes. The mixed content code is emitted as an optional sidecar under `content_codes["CONTENT_MIXED_V0"]`, not folded into the default composite ISCC-CODE. This preserves the baseline Data/Instance semantics and follows the ISCC constraint that a composite code has a single Content subtype.

For each scene, `iscc_bio.biocode._selected_mixed_content_code`:

1. uses the existing IMAGEWALK order;
2. selects all planes for scenes with up to three planes, otherwise first, middle, and last offsets;
3. normalizes each selected 2D plane to 32 × 32 grayscale `uint8` using deterministic percentile clipping and bilinear resize;
4. generates standard image Content-Codes with `iscc_lib.gen_image_code_v0(..., bits=256)`;
5. combines them with `iscc_lib.gen_mixed_code_v0(..., bits=256)` into a standard Content-Code-Mixed unit;
6. records the mixed unit, selected offsets, selected count, source plane count, bit length, and per-plane image Content-Codes.

`gen_mixed_code_v0` requires at least two Content-Codes, so one-plane scenes duplicate the one selected Image Content-Code internally while reporting the actual selected offset only once.

## Verification result

The generated JOSS experiment now computes the optional sidecar for original and converted scene codes and reports Content-Code-Mixed equality/Hamming distance in `results.csv`, `results.json`, and `paper/experiment-results.md`.

Observed behavior on the current corpus:

- exact `tifffile` OME-TIFF round trips: mixed code equal for all six decoded samples;
- DEFLATE-compressed OME-TIFF round trips: mixed code equal for all six decoded samples;
- exact `ome-zarr-py` OME-Zarr round trips: mixed code equal for all six decoded samples;
- one-pixel drift variants: mixed code equal for all twelve OME-TIFF/OME-Zarr drift rows;
- `bfconvert` exact rows: mixed code equal for the four rows that were exact Data/Instance matches;
- problematic `bfconvert` CZI/LIF rows: mixed code detects differences rather than hiding reader/converter divergence.

The current summary is 34 rows with exact Content-Code-Mixed equality out of 40 total experiment rows. The remaining rows are the OIR reader skip, CZI/LIF `bfconvert` mismatches, and LIF scene-count mismatch rows.

## Verdict: IMPLEMENTED AS OPTIONAL SIDECAR

The sidecar is robust for the conversion classes this JOSS experiment is designed to exercise: lossless container compression and tiny decoded-pixel drift. It is intentionally reported at the Content layer only. Instance-Code remains exact-identity and Data-Code remains byte/canonical-stream similarity; the mixed Content-Code does not make those units compression-tolerant.

## Follow-up

A larger precision/recall benchmark, ideally with naturally occurring lossy microscopy codec drift, is still needed before promoting Content-Code-Mixed sidecars as a default retrieval feature or claiming broad lossy-compression robustness.
