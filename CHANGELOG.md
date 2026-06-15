# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for Python 3.14 (tested in CI alongside 3.11–3.13)

### Changed

- Update all dependencies to latest versions (including optional reader plugins and dev tooling)
- Rename the `ome-zarr-plugin` extra to `ome-zarr` for naming consistency with other format extras
    (**breaking**: install with `iscc-bio[ome-zarr]` instead of `iscc-bio[ome-zarr-plugin]`)

## [0.1.0] - 2025-05-12

- Initial release
