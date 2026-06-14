# Mixed Content-Code spike

## Question

Would an additional ISCC Content-Code-Mixed unit, generated from a deterministic subset of IMAGEWALK planes, improve `iscc-bio` robustness enough to add to the default BioCode output and JOSS experiment?

## Approach

A throwaway spike under `/tmp/iscc_bio_mixed_content_spike.py` tested the current JOSS conversion artifacts by:

1. grouping planes in IMAGEWALK order;
2. selecting three evenly spaced plane positions per scene;
3. normalizing each selected 2D plane to 32 × 32 grayscale `uint8`;
4. generating standard image Content-Codes with `iscc_lib.gen_image_code_v0(..., bits=256)`;
5. combining the selected plane codes with `iscc_lib.gen_mixed_code_v0(..., bits=256)`;
6. comparing mixed-code bodies by Hamming distance across original and converted scenes.

The design is aligned with ISCC primitives: several per-plane image Content-Codes must be collapsed into a single Content-Code-Mixed unit before composition, because a composite ISCC-CODE may contain at most one Content unit.

## Result

The spike was technically feasible with the installed `iscc-lib` API and preserved the mixed-code `parts` list as expected.

Observed behavior on the current conversion artifacts:

- exact `tifffile` OME-TIFF round trips: mixed code equal for all tested scenes;
- exact DEFLATE OME-TIFF round trips: mixed code equal for all tested scenes;
- exact `ome-zarr-py` OME-Zarr round trips: mixed code equal for all tested scenes;
- one-pixel drift variants: mixed code remained equal for all tested scenes;
- problematic `bfconvert` CZI/LIF cases: mixed code also detected differences rather than rescuing them into near matches.

## Verdict: PARTIAL

The mixed Content-Code is deterministic and stable for the current exact-conversion and one-pixel-drift cases, but the spike did not show enough incremental value over the existing Data-Code/Instance-Code behavior to justify adding it to the default output path for the JOSS paper.

## Recommendation

Defer production integration until a larger validation set can measure retrieval precision/recall across realistic biological imaging perturbations.

If implemented later, make it optional and explicit, for example `--content-mixed`, and preserve:

- the mixed Content-Code unit;
- the per-plane image Content-Code `parts`;
- selected plane indices and `(z, c, t)` coordinates;
- the exact selection policy, e.g. `imagewalk_evenly_spaced_3_v0`.

For one-plane scenes, prefer reporting a single per-plane image Content-Code or omitting the mixed unit rather than duplicating the same plane three times, unless fixed-cardinality output is required.
