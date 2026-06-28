# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Top-level raw-byte ISCC-SUM for `biocode()` over the original source — a single file or a directory tree (e.g.
    an OME-Zarr store, traversed in TREEWALK-ISCC order) — with the per-scene IMAGEWALK codes nested as `parts`.
    The top-level code is decode-independent (still produced if pixel reading fails) and, for a single file,
    identical to the standard `iscc-sum` tool (`iscc_bio/rawsum.py`)
- `iscc_bio/treewalk.py` — deterministic, cross-platform directory tree traversal (TREEWALK-BASE/IGNORE/ISCC)
    ported from `bio-codes/iscc-sum`, for reproducible raw-byte tree ISCC-SUMs over multi-file bioimages
- Support for Python 3.14 (tested in CI alongside 3.11–3.13)

### Changed

- **Breaking**: `iscc_bio.api.biocode()` now returns a single container dict (`iscc_code`, `units`, `datahash`,
    `filesize`, `generator`, `parts`) instead of a list of per-scene entries; the former per-scene list is now
    `result["parts"]`
- Require `iscc-lib>=0.5.0`, which releases the GIL during Data/Instance hashing, enabling thread-level
    parallelism for hashing-heavy workloads
- Update all dependencies to latest versions (including optional reader plugins and dev tooling)
- Rename the `ome-zarr-plugin` extra to `ome-zarr` for naming consistency with other format extras
    (**breaking**: install with `iscc-bio[ome-zarr]` instead of `iscc-bio[ome-zarr-plugin]`)

## [0.1.0] - 2025-05-12

- Initial release
