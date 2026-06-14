#!/usr/bin/env python3
"""Reproducible conversion and near-match experiment for the JOSS paper.

The experiment downloads a small public bioimage corpus, converts each readable
sample into several normalized variants, and evaluates two complementary
``iscc-bio`` claims:

* exact IMAGEWALK round trips: conversions that preserve canonical planes should
  keep both Data-Code and Instance-Code units identical;
* slight canonical-data drift: conversions that introduce a small pixel-level
  inconsistency should change the Instance-Code while keeping the Data-Code close
  enough to be retrieved by Hamming-distance search.

Outputs are JSON/CSV artifacts under ``experiments/results/`` plus a tracked
paper table and PNG figure under ``paper/`` by default. The sample cache and
converted files are ignored by git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import iscc_lib
import numpy as np
import tifffile
import zarr
from ome_zarr.writer import write_image

from iscc_bio.api import biocode
from iscc_bio.imagewalk import iter_planes_bioio
from iscc_bio.imagewalk.common import Plane


MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_TOOL_DOWNLOAD_BYTES = 100 * 1024 * 1024
LARGE_TOOL_DOWNLOAD_BYTES = 450 * 1024 * 1024
DEFAULT_CACHE = Path("experiments/cache/joss-bioimages")
DEFAULT_TOOL_CACHE = Path("experiments/cache/joss-tools")
DEFAULT_RESULTS = Path("experiments/results/joss-conversion-matching")
DEFAULT_PAPER_TABLE = Path("paper/experiment-results.md")
DEFAULT_PAPER_FIGURE = Path("paper/figures/conversion-matching.png")
DEFAULT_DATA_HAMMING_THRESHOLD = 64
DEFAULT_MAX_SCENE_BYTES = 64 * 1024 * 1024
DEFAULT_EXTERNAL_TIMEOUT = 600
ISCC_UNIT_BITS = 256


@dataclass(frozen=True)
class Sample:
    """Public corpus entry."""

    sample_id: str
    url: str
    expected_size: int
    sha256: str
    source: str
    format: str
    note: str


@dataclass(frozen=True)
class ToolArchive:
    """Pinned downloadable command-line tool archive."""

    tool_id: str
    label: str
    version: str
    url: str
    expected_size: int
    sha256: str
    executables: tuple[str, ...]
    default_enabled: bool
    large_download: bool
    note: str


@dataclass(frozen=True)
class ToolState:
    """Resolution state for an optional command-line tool archive."""

    tool_id: str
    label: str
    version: str
    status: str
    executable: str = ""
    version_output: str = ""
    error: str = ""


PUBLIC_CORPUS: tuple[Sample, ...] = (
    Sample(
        sample_id="ome_tiff_multi_channel_4d",
        url="https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-4D-series.ome.tiff",
        expected_size=7889665,
        sha256="23ec5b84154850360800b299e6c088b8f60c5e81b6c990ac1e9b15496fa9549d",
        source="Open Microscopy sample images",
        format="OME-TIFF",
        note="Artificial multi-channel 4D OME-TIFF fixture.",
    ),
    Sample(
        sample_id="tiff_condensation_c4",
        url="https://downloads.openmicroscopy.org/images/TIFF/condensation/C4.pattern1.tif",
        expected_size=6749325,
        sha256="d3d68108d3d9155fef16971bc9754967ec7ce03728dae91ebf8c5ac152f32d18",
        source="Open Microscopy TIFF condensation sample",
        format="TIFF",
        note="Condensation TIFF sample with pinned size and digest.",
    ),
    Sample(
        sample_id="zeiss_czi_rgb_8bit",
        url="https://raw.githubusercontent.com/bioio-devs/aicspylibczi/facc78428403e0cc1c5eea0f51f47e22dfffc0bf/aicspylibczi/tests/resources/RGB-8bit.czi",
        expected_size=1749952,
        sha256="44593e6210f2f9066f8608c53c31806f4d173d1d26bb3bb5c32b182fb0c0a43e",
        source="bioio-devs/aicspylibczi test resources",
        format="CZI",
        note="Small Zeiss CZI RGB fixture.",
    ),
    Sample(
        sample_id="nikon_nd2_bf007",
        url="https://downloads.openmicroscopy.org/images/ND2/maxime/BF007.nd2",
        expected_size=270336,
        sha256="e3f449796ac2ce82d5734a607d8899a3ec5a9a575161a3eded2e22eedc1dce92",
        source="Open Microscopy Nikon ND2 sample",
        format="ND2",
        note="Small Nikon NIS-Elements ND2 fixture.",
    ),
    Sample(
        sample_id="olympus_oir_map_a01",
        url="https://downloads.openmicroscopy.org/images/Olympus-OIR/gh-4205/zenodo-13680725/Map_A01.oir",
        expected_size=1701149,
        sha256="a5d5cdfb8401da9d47586150ca7edab8f61b70aa50e4d36f148bc8cdbf703188",
        source="Open Microscopy Olympus OIR sample",
        format="OIR",
        note="Olympus/Evident OIR example; requires a compatible BioIO reader.",
    ),
    Sample(
        sample_id="leica_lif_pr2729",
        url="https://downloads.openmicroscopy.org/images/Leica-LIF/michael/PR2729_frameOrderCombinedScanTypes.lif",
        expected_size=227429,
        sha256="17994bb1bdf93dd1a34a19ed53e5a3d7f672470a8057e2998e25d2474437a052",
        source="Open Microscopy Leica LIF sample",
        format="LIF",
        note="Small Leica LIF example; requires a compatible BioIO reader.",
    ),
)


TOOL_ARCHIVES: tuple[ToolArchive, ...] = (
    ToolArchive(
        tool_id="bftools",
        label="Bio-Formats command-line tools",
        version="8.5.0",
        url="https://downloads.openmicroscopy.org/bio-formats/8.5.0/artifacts/bftools.zip",
        expected_size=51360836,
        sha256="07a3bb1d3de84da3a709655a1008cb2d9b19becc5bad4ae4112633aec9380478",
        executables=("bfconvert", "bfconvert.bat"),
        default_enabled=True,
        large_download=False,
        note="Provides pinned bfconvert used by the default paper experiment.",
    ),
    ToolArchive(
        tool_id="bioformats2raw",
        label="bioformats2raw",
        version="0.12.0",
        url="https://github.com/glencoesoftware/bioformats2raw/releases/download/v0.12.0/bioformats2raw-0.12.0.zip",
        expected_size=205038978,
        sha256="82964c3e2e4b5f27e3bafbd7fdca2afe4570f44863d2363f7d0ba2a23b9d39e3",
        executables=("bioformats2raw", "bioformats2raw.bat"),
        default_enabled=False,
        large_download=True,
        note="Gated optional raw-pyramid leg; not downloaded by the default run.",
    ),
    ToolArchive(
        tool_id="raw2ometiff",
        label="raw2ometiff",
        version="0.10.0",
        url="https://github.com/glencoesoftware/raw2ometiff/releases/download/v0.10.0/raw2ometiff-0.10.0.zip",
        expected_size=205069533,
        sha256="a9efe3669a853a698bc7a0ce3723df44cb66ae3aab739aeddea434f7fd3002ba",
        executables=("raw2ometiff", "raw2ometiff.bat"),
        default_enabled=False,
        large_download=True,
        note="Gated optional OME-TIFF leg after bioformats2raw.",
    ),
)


EXCLUDED_CONVERTERS: tuple[dict[str, str], ...] = (
    {
        "tool": "Bisque bioimage-convert / imgcnv",
        "status": "considered/excluded",
        "reason": (
            "Current public infrastructure for a pinned, scriptable imgcnv run is "
            "unavailable or not reproducible for this experiment."
        ),
    },
)


@dataclass(frozen=True)
class Converter:
    """A concrete conversion condition used by the experiment."""

    variant: str
    tool: str
    target_format: str
    drift_pixels: int
    source_type: str
    suffix: str
    note: str
    scope: str = "scene"
    writer: Callable[[np.ndarray, Path], None] | None = None
    external_tool_ids: tuple[str, ...] = ()


@dataclass
class MatchRow:
    sample_id: str
    scene_idx: int
    variant: str
    tool: str
    target_format: str
    drift_pixels: int
    status: str
    scene_count: int = 0
    matching_scenes: int = 0
    data_near_match: bool = False
    data_code_equal: bool = False
    instance_code_equal: bool = False
    data_hamming: int = -1
    instance_hamming: int = -1
    data_bits: int = ISCC_UNIT_BITS
    instance_bits: int = ISCC_UNIT_BITS
    data_hamming_threshold: int = DEFAULT_DATA_HAMMING_THRESHOLD
    shape_tczyx: str = ""
    dtype: str = ""
    plane_count: int = 0
    original_path: str = ""
    variant_path: str = ""
    error: str = ""


@dataclass
class SceneBundle:
    scene_idx: int
    planes: list[Plane]


@dataclass(frozen=True)
class CodeComparison:
    status: str
    matching_scenes: int
    scene_count: int
    data_near_match: bool
    data_code_equal: bool
    instance_code_equal: bool
    data_hamming: int
    instance_hamming: int
    data_bits: int
    instance_bits: int


def sample_filename(sample: Sample) -> str:
    """Return a deterministic local filename for a sample URL."""

    suffix = Path(urllib.request.url2pathname(sample.url.rsplit("/", 1)[-1])).suffix
    if sample.url.endswith(".ome.tiff"):
        suffix = ".ome.tiff"
    if not suffix:
        suffix = ".bin"
    return f"{sample.sample_id}{suffix}"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_sample(sample: Sample, cache_dir: Path, *, offline: bool) -> Path:
    """Download ``sample`` into ``cache_dir`` unless it is already cached."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / sample_filename(sample)
    if target.exists():
        verify_sample_file(sample, target)
        return target
    if offline:
        raise FileNotFoundError(f"{target} is not cached and --offline was requested")
    if sample.expected_size > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"refusing to download {sample.sample_id}: expected size {sample.expected_size} exceeds {MAX_DOWNLOAD_BYTES}"
        )

    with urllib.request.urlopen(sample.url, timeout=60) as response:  # noqa: S310 - fixed public sample URLs
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"refusing to download {sample.sample_id}: server reports {length} bytes"
            )
        tmp = target.with_suffix(target.suffix + ".tmp")
        total = 0
        with tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(
                        f"refusing to keep {sample.sample_id}: downloaded over {MAX_DOWNLOAD_BYTES} bytes"
                    )
                out.write(chunk)
        tmp.replace(target)
    verify_sample_file(sample, target)
    return target


def verify_sample_file(sample: Sample, path: Path) -> None:
    """Verify cached public sample size and SHA-256 digest."""

    actual_size = path.stat().st_size
    if actual_size != sample.expected_size:
        raise ValueError(
            f"{sample.sample_id}: expected {sample.expected_size} bytes but found {actual_size}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != sample.sha256:
        raise ValueError(
            f"{sample.sample_id}: expected sha256 {sample.sha256} but found {actual_sha256}"
        )


def verify_downloaded_file(
    path: Path, *, label: str, expected_size: int, expected_sha256: str
) -> None:
    """Verify a downloaded file size and SHA-256 digest."""

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{label}: expected {expected_size} bytes but found {actual_size}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label}: expected sha256 {expected_sha256} but found {actual_sha256}"
        )


def download_verified_file(
    *,
    url: str,
    target: Path,
    label: str,
    expected_size: int,
    expected_sha256: str,
    offline: bool,
    max_download_bytes: int,
) -> Path:
    """Download a pinned URL into ``target`` and verify size and SHA-256."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify_downloaded_file(
            target,
            label=label,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        return target
    if offline:
        raise FileNotFoundError(f"{target} is not cached and --offline was requested")
    if expected_size > max_download_bytes:
        raise ValueError(
            f"refusing to download {label}: expected size {expected_size} exceeds {max_download_bytes}"
        )

    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed public URLs
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_download_bytes:
            raise ValueError(
                f"refusing to download {label}: server reports {length} bytes"
            )
        tmp = target.with_suffix(target.suffix + ".tmp")
        total = 0
        with tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_download_bytes:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(
                        f"refusing to keep {label}: downloaded over {max_download_bytes} bytes"
                    )
                out.write(chunk)
        tmp.replace(target)
    verify_downloaded_file(
        target,
        label=label,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    return target


def executable_sort_key(path: Path) -> tuple[int, str]:
    """Prefer POSIX launchers over Windows batch files on POSIX systems."""

    return (1 if path.suffix.lower() == ".bat" else 0, str(path))


def find_tool_executable(extract_dir: Path, executable_names: Sequence[str]) -> Path:
    """Find an executable from an extracted tool archive."""

    candidates: list[Path] = []
    for name in executable_names:
        candidates.extend(path for path in extract_dir.rglob(name) if path.is_file())
    if not candidates:
        names = ", ".join(executable_names)
        raise FileNotFoundError(f"no executable named {names} below {extract_dir}")
    executable = sorted(candidates, key=executable_sort_key)[0]
    try:
        executable.chmod(executable.stat().st_mode | 0o111)
    except OSError:
        pass
    return executable


def extract_tool_archive(archive: Path, extract_dir: Path, *, marker_name: str) -> None:
    """Extract a verified ZIP archive once, replacing stale extracted contents."""

    marker = extract_dir / marker_name
    if marker.exists():
        return
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(extract_dir)
    for path in extract_dir.rglob("*"):
        if path.is_file() and (
            path.suffix in {"", ".sh"} or path.parent.name == "bftools"
        ):
            try:
                path.chmod(path.stat().st_mode | 0o111)
            except OSError:
                pass
    marker.write_text(archive.name + "\n")


def command_version(executable: Path) -> str:
    """Return a short version string for an external command."""

    fallback = "installed"
    for flag in ("--version", "-version"):
        try:
            completed = subprocess.run(  # noqa: S603 - fixed executable path
                [str(executable), flag],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            continue
        output = (completed.stdout or completed.stderr).strip().splitlines()
        if completed.returncode == 0:
            return output[0] if output else "installed"
        if output and fallback == "installed":
            fallback = output[0]
    return fallback


def resolve_tool_archive(
    spec: ToolArchive,
    tool_cache_dir: Path,
    *,
    offline: bool,
    max_tool_download_bytes: int,
) -> ToolState:
    """Resolve a pinned tool archive into an executable, returning errors as state."""

    try:
        archive = download_verified_file(
            url=spec.url,
            target=tool_cache_dir / "downloads" / f"{spec.tool_id}-{spec.version}.zip",
            label=f"{spec.tool_id} {spec.version}",
            expected_size=spec.expected_size,
            expected_sha256=spec.sha256,
            offline=offline,
            max_download_bytes=max_tool_download_bytes,
        )
        extract_dir = tool_cache_dir / f"{spec.tool_id}-{spec.version}"
        extract_tool_archive(
            archive,
            extract_dir,
            marker_name=f".extracted-{spec.sha256[:16]}",
        )
        executable = find_tool_executable(extract_dir, spec.executables)
        return ToolState(
            tool_id=spec.tool_id,
            label=spec.label,
            version=spec.version,
            status="ready",
            executable=str(executable),
            version_output=command_version(executable),
        )
    except Exception as exc:
        return ToolState(
            tool_id=spec.tool_id,
            label=spec.label,
            version=spec.version,
            status="error",
            error=repr(exc),
        )


def requested_tool_ids(mode: str) -> set[str]:
    """Return tool archive identifiers requested by an external-tool mode."""

    if mode == "none":
        return set()
    if mode == "bftools":
        return {"bftools"}
    if mode == "all":
        return {spec.tool_id for spec in TOOL_ARCHIVES}
    raise ValueError(f"unknown external tool mode: {mode!r}")


def resolve_external_tools(args: argparse.Namespace) -> dict[str, ToolState]:
    """Resolve optional external converters according to CLI gating."""

    requested = requested_tool_ids(args.external_tools)
    max_download = (
        LARGE_TOOL_DOWNLOAD_BYTES
        if args.allow_large_tool_downloads
        else args.max_tool_download_bytes
    )
    states: dict[str, ToolState] = {}
    for spec in TOOL_ARCHIVES:
        if spec.tool_id not in requested:
            states[spec.tool_id] = ToolState(
                tool_id=spec.tool_id,
                label=spec.label,
                version=spec.version,
                status="disabled",
            )
            continue
        if spec.large_download and not args.allow_large_tool_downloads:
            states[spec.tool_id] = ToolState(
                tool_id=spec.tool_id,
                label=spec.label,
                version=spec.version,
                status="gated",
                error=(
                    "large optional archive; pass --allow-large-tool-downloads "
                    "with --external-tools all to enable"
                ),
            )
            continue
        states[spec.tool_id] = resolve_tool_archive(
            spec,
            args.tool_cache_dir,
            offline=args.offline,
            max_tool_download_bytes=max_download,
        )
    return states


def require_ready_tool(tools: dict[str, ToolState], tool_id: str) -> Path:
    """Return a resolved tool executable or raise a clear error."""

    state = tools[tool_id]
    if state.status != "ready" or not state.executable:
        detail = state.error or state.status
        raise RuntimeError(f"{state.label} {state.version} is not ready: {detail}")
    return Path(state.executable)


def group_by_scene(planes: Iterable[Plane]) -> list[SceneBundle]:
    """Materialize planes from an IMAGEWALK iterator grouped by scene."""

    scenes: dict[int, list[Plane]] = {}
    for plane in planes:
        scenes.setdefault(plane.scene_idx, []).append(plane)
    return [SceneBundle(scene_idx=idx, planes=scenes[idx]) for idx in sorted(scenes)]


def scene_to_tczyx(scene: SceneBundle) -> np.ndarray:
    """Return a scene as a dense TCZYX array for OME writers."""

    if not scene.planes:
        raise ValueError(f"scene {scene.scene_idx} has no planes")

    t_size = max(plane.t_time for plane in scene.planes) + 1
    c_size = max(plane.c_channel for plane in scene.planes) + 1
    z_size = max(plane.z_depth for plane in scene.planes) + 1
    y_size, x_size = scene.planes[0].xy_array.shape
    dtype = scene.planes[0].xy_array.dtype
    array = np.zeros((t_size, c_size, z_size, y_size, x_size), dtype=dtype)

    for plane in scene.planes:
        if plane.xy_array.shape != (y_size, x_size):
            raise ValueError("all planes in a scene must share YX shape for conversion")
        if plane.xy_array.dtype != dtype:
            raise ValueError("all planes in a scene must share dtype for conversion")
        array[plane.t_time, plane.c_channel, plane.z_depth] = plane.xy_array
    return array


def perturb_array(array_tczyx: np.ndarray, *, pixels: int = 1) -> np.ndarray:
    """Return a copy with deterministic one-level pixel perturbations.

    This simulates the small decoded-pixel drift seen in some lossy or
    decoder-dependent conversions (for example CZI/JPEG-XR pipelines) without
    depending on a large or restricted public fixture.
    """

    if pixels < 0:
        raise ValueError("pixels must be non-negative")
    drifted = array_tczyx.copy()
    if pixels == 0 or drifted.size == 0:
        return drifted

    flat = drifted.reshape(-1)
    for idx in range(min(pixels, flat.size)):
        value = flat[idx]
        if np.issubdtype(drifted.dtype, np.integer):
            info = np.iinfo(drifted.dtype)
            flat[idx] = value + 1 if value < info.max else value - 1
        elif np.issubdtype(drifted.dtype, np.floating):
            flat[idx] = np.nextafter(value, np.inf, dtype=drifted.dtype)
        else:
            raise TypeError(f"unsupported dtype for perturbation: {drifted.dtype}")
    return drifted


def write_ome_tiff(array_tczyx: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, array_tczyx, ome=True, metadata={"axes": "TCZYX"})


def write_ome_tiff_deflate(array_tczyx: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output,
        array_tczyx,
        ome=True,
        compression="deflate",
        metadata={"axes": "TCZYX"},
    )


def write_ome_zarr(array_tczyx: np.ndarray, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output), mode="w", zarr_format=2)
    # ``scale_factors=[]`` keeps the fixture single-scale and fast. Axes are
    # explicit so ``iter_planes_ngff`` can reconstruct T/C/Z/Y/X positions.
    write_image(array_tczyx, root, axes="tczyx", scale_factors=[])


def remove_output_path(path: Path) -> None:
    """Remove a previous converter output path."""

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def run_external_command(command: Sequence[str], *, timeout: int) -> None:
    """Run an external converter and include captured output on failure."""

    completed = subprocess.run(  # noqa: S603 - fixed executable paths and args
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
        raise RuntimeError(
            f"external command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{output}"
        )


def run_bfconvert(
    source_path: Path,
    output: Path,
    *,
    tools: dict[str, ToolState],
    timeout: int,
) -> None:
    """Convert a source file to OME-TIFF with pinned Bio-Formats bfconvert."""

    bfconvert = require_ready_tool(tools, "bftools")
    output.parent.mkdir(parents=True, exist_ok=True)
    remove_output_path(output)
    run_external_command(
        [str(bfconvert), str(source_path), str(output)], timeout=timeout
    )


def run_bioformats2raw_raw2ometiff(
    source_path: Path,
    output: Path,
    *,
    tools: dict[str, ToolState],
    timeout: int,
) -> None:
    """Convert source -> raw Zarr -> OME-TIFF with the gated raw pipeline."""

    bioformats2raw = require_ready_tool(tools, "bioformats2raw")
    raw2ometiff = require_ready_tool(tools, "raw2ometiff")
    output.parent.mkdir(parents=True, exist_ok=True)
    remove_output_path(output)
    raw_zarr = output.with_name(output.name + ".raw.zarr")
    remove_output_path(raw_zarr)
    try:
        run_external_command(
            [str(bioformats2raw), str(source_path), str(raw_zarr)], timeout=timeout
        )
        run_external_command(
            [str(raw2ometiff), str(raw_zarr), str(output)], timeout=timeout
        )
    finally:
        remove_output_path(raw_zarr)


def run_source_file_converter(
    converter: Converter,
    source_path: Path,
    output: Path,
    *,
    tools: dict[str, ToolState],
    timeout: int,
) -> None:
    """Run an external source-file converter."""

    if converter.variant == "ome_tiff_bfconvert":
        run_bfconvert(source_path, output, tools=tools, timeout=timeout)
        return
    if converter.variant == "ome_tiff_bioformats2raw_raw2ometiff":
        run_bioformats2raw_raw2ometiff(
            source_path, output, tools=tools, timeout=timeout
        )
        return
    raise ValueError(f"unknown source-file converter: {converter.variant}")


def converter_registry(external_tools: str = "bftools") -> tuple[Converter, ...]:
    """Return the built-in conversion conditions.

    The scene-scoped converters are exact round trips through Python writer
    stacks and target bioimage storage formats. Source-scoped converters run
    pinned external command-line tools against the original public sample.
    Drift variants deliberately apply a one-pixel perturbation before writing
    to validate Data-Code near matching when Instance-Code equality fails.
    """

    converters = [
        Converter(
            variant="ome_tiff_tifffile",
            tool="tifffile",
            target_format="OME-TIFF",
            drift_pixels=0,
            source_type="bioio",
            suffix=".ome.tiff",
            note="OME-TIFF written by tifffile with explicit TCZYX axes.",
            writer=write_ome_tiff,
        ),
        Converter(
            variant="ome_tiff_tifffile_deflate",
            tool="tifffile(deflate)",
            target_format="OME-TIFF",
            drift_pixels=0,
            source_type="bioio",
            suffix=".deflate.ome.tiff",
            note="Compressed OME-TIFF written by tifffile/DEFLATE.",
            writer=write_ome_tiff_deflate,
        ),
        Converter(
            variant="ome_zarr_ome_zarr_py",
            tool="ome-zarr-py",
            target_format="OME-Zarr",
            drift_pixels=0,
            source_type="zarr",
            suffix=".zarr",
            note="Single-scale OME-NGFF/Zarr written by ome-zarr-py.",
            writer=write_ome_zarr,
        ),
        Converter(
            variant="ome_tiff_tifffile_one_pixel_drift",
            tool="tifffile+synthetic-drift",
            target_format="OME-TIFF",
            drift_pixels=1,
            source_type="bioio",
            suffix=".drift.ome.tiff",
            note="OME-TIFF after deterministic one-pixel +1 perturbation.",
            writer=write_ome_tiff,
        ),
        Converter(
            variant="ome_zarr_ome_zarr_py_one_pixel_drift",
            tool="ome-zarr-py+synthetic-drift",
            target_format="OME-Zarr",
            drift_pixels=1,
            source_type="zarr",
            suffix=".drift.zarr",
            note="OME-Zarr after deterministic one-pixel +1 perturbation.",
            writer=write_ome_zarr,
        ),
    ]
    if "bftools" in requested_tool_ids(external_tools):
        converters.append(
            Converter(
                variant="ome_tiff_bfconvert",
                tool="bfconvert 8.5.0",
                target_format="OME-TIFF",
                drift_pixels=0,
                source_type="bioio",
                suffix=".bfconvert.ome.tiff",
                note="OME-TIFF written from the original source by pinned Bio-Formats bfconvert.",
                scope="sample",
                external_tool_ids=("bftools",),
            )
        )
    if {"bioformats2raw", "raw2ometiff"} <= requested_tool_ids(external_tools):
        converters.append(
            Converter(
                variant="ome_tiff_bioformats2raw_raw2ometiff",
                tool="bioformats2raw 0.12.0 + raw2ometiff 0.10.0",
                target_format="OME-TIFF",
                drift_pixels=0,
                source_type="bioio",
                suffix=".raw2ometiff.ome.tiff",
                note=(
                    "Gated source -> OME-NGFF/Zarr -> OME-TIFF pipeline using "
                    "pinned GLencoe command-line tools."
                ),
                scope="sample",
                external_tool_ids=("bioformats2raw", "raw2ometiff"),
            )
        )
    return tuple(converters)


def code_for_path(path: Path, *, source_type: str) -> list[dict]:
    return biocode(path, source_type=source_type, simprints=False, bits=ISCC_UNIT_BITS)


def code_identity(entry: dict) -> tuple[str, list[str]]:
    """Return the BioCode identity fields or fail loudly on schema drift."""

    try:
        iscc_code = entry["iscc_code"]
        units = entry["units"]
    except KeyError as exc:
        raise KeyError(
            f"biocode entry missing required key {exc.args[0]!r}: {entry!r}"
        ) from exc
    if (
        not isinstance(iscc_code, str)
        or not isinstance(units, list)
        or len(units) != 2
        or not all(isinstance(unit, str) for unit in units)
    ):
        raise TypeError(f"unexpected biocode entry shape: {entry!r}")
    return iscc_code, units


def unit_body(code: str) -> bytes:
    """Return the decoded ISCC unit body bytes."""

    return iscc_lib.iscc_decode(code)[4]


def hamming_distance(left: bytes, right: bytes) -> int:
    """Return bit-level Hamming distance for equal-length byte strings."""

    if len(left) != len(right):
        raise ValueError(f"cannot compare {len(left)} and {len(right)} bytes")
    return sum((a ^ b).bit_count() for a, b in zip(left, right, strict=True))


def compare_entries(
    left_entry: dict, right_entry: dict, *, data_hamming_threshold: int
) -> CodeComparison:
    """Compare two scene-level ISCC-SUM entries component-wise."""

    left_sum, left_units = code_identity(left_entry)
    right_sum, right_units = code_identity(right_entry)
    left_data, left_instance = left_units
    right_data, right_instance = right_units

    data_hamming = hamming_distance(unit_body(left_data), unit_body(right_data))
    instance_hamming = hamming_distance(
        unit_body(left_instance), unit_body(right_instance)
    )
    data_bits = len(unit_body(left_data)) * 8
    instance_bits = len(unit_body(left_instance)) * 8
    data_code_equal = left_data == right_data
    instance_code_equal = left_instance == right_instance
    data_near_match = data_hamming <= data_hamming_threshold

    if left_sum == right_sum and data_code_equal and instance_code_equal:
        status = "exact_match"
        matching = 1
    elif data_near_match and not instance_code_equal:
        status = "near_match"
        matching = 0
    else:
        status = "mismatch"
        matching = 0

    return CodeComparison(
        status=status,
        matching_scenes=matching,
        scene_count=1,
        data_near_match=data_near_match,
        data_code_equal=data_code_equal,
        instance_code_equal=instance_code_equal,
        data_hamming=data_hamming,
        instance_hamming=instance_hamming,
        data_bits=data_bits,
        instance_bits=instance_bits,
    )


def scene_codes_match(left: Sequence[dict], right: Sequence[dict]) -> tuple[int, int]:
    """Backward-compatible exact scene-code comparison for tests."""

    if len(left) != len(right):
        return 0, max(len(left), len(right))

    matching = 0
    total = len(left)
    for left_entry, right_entry in zip(left, right, strict=True):
        if code_identity(left_entry) == code_identity(right_entry):
            matching += 1
    return matching, total


def scene_metadata(scene: SceneBundle | None) -> tuple[str, str, int]:
    """Return shape, dtype, and plane count metadata for a scene."""

    if scene is None or not scene.planes:
        return "", "", 0
    t_size = max(plane.t_time for plane in scene.planes) + 1
    c_size = max(plane.c_channel for plane in scene.planes) + 1
    z_size = max(plane.z_depth for plane in scene.planes) + 1
    y_size, x_size = scene.planes[0].xy_array.shape
    shape = "x".join(str(part) for part in (t_size, c_size, z_size, y_size, x_size))
    return shape, str(scene.planes[0].xy_array.dtype), len(scene.planes)


def comparison_row(
    *,
    sample: Sample,
    scene_idx: int,
    converter: Converter,
    comparison: CodeComparison,
    data_hamming_threshold: int,
    shape: str,
    dtype: str,
    plane_count: int,
    original_path: Path,
    variant_path: Path,
) -> MatchRow:
    """Build a result row from a successful code comparison."""

    return MatchRow(
        sample_id=sample.sample_id,
        scene_idx=scene_idx,
        variant=converter.variant,
        tool=converter.tool,
        target_format=converter.target_format,
        drift_pixels=converter.drift_pixels,
        status=comparison.status,
        scene_count=comparison.scene_count,
        matching_scenes=comparison.matching_scenes,
        data_near_match=comparison.data_near_match,
        data_code_equal=comparison.data_code_equal,
        instance_code_equal=comparison.instance_code_equal,
        data_hamming=comparison.data_hamming,
        instance_hamming=comparison.instance_hamming,
        data_bits=comparison.data_bits,
        instance_bits=comparison.instance_bits,
        data_hamming_threshold=data_hamming_threshold,
        shape_tczyx=shape,
        dtype=dtype,
        plane_count=plane_count,
        original_path=str(original_path),
        variant_path=str(variant_path),
    )


def converter_error_row(
    *,
    sample: Sample,
    scene_idx: int,
    converter: Converter,
    data_hamming_threshold: int,
    original_path: Path,
    error: str,
    variant_path: Path | None = None,
    shape: str = "",
    dtype: str = "",
    plane_count: int = 0,
    status: str = "error",
) -> MatchRow:
    """Build a skip/error/mismatch result row for a converter."""

    return MatchRow(
        sample_id=sample.sample_id,
        scene_idx=scene_idx,
        variant=converter.variant,
        tool=converter.tool,
        target_format=converter.target_format,
        drift_pixels=converter.drift_pixels,
        status=status,
        data_hamming_threshold=data_hamming_threshold,
        shape_tczyx=shape,
        dtype=dtype,
        plane_count=plane_count,
        original_path=str(original_path),
        variant_path=str(variant_path or ""),
        error=error,
    )


def evaluate_source_file_converter(
    sample: Sample,
    source_path: Path,
    scenes: Sequence[SceneBundle],
    original_codes: Sequence[dict],
    work_dir: Path,
    converter: Converter,
    *,
    tools: dict[str, ToolState],
    external_timeout: int,
    data_hamming_threshold: int,
) -> Iterator[MatchRow]:
    """Run one external source-file converter and compare all resulting scenes."""

    path = work_dir / sample.sample_id / f"{sample.sample_id}{converter.suffix}"
    try:
        run_source_file_converter(
            converter,
            source_path,
            path,
            tools=tools,
            timeout=external_timeout,
        )
        variant_codes = code_for_path(path, source_type=converter.source_type)
    except Exception as exc:
        scene_count = max(len(scenes), 1)
        for scene_idx in range(scene_count):
            scene = scenes[scene_idx] if scene_idx < len(scenes) else None
            shape, dtype, plane_count = scene_metadata(scene)
            yield converter_error_row(
                sample=sample,
                scene_idx=scene.scene_idx if scene is not None else -1,
                converter=converter,
                data_hamming_threshold=data_hamming_threshold,
                shape=shape,
                dtype=dtype,
                plane_count=plane_count,
                original_path=source_path,
                variant_path=path,
                error=repr(exc),
            )
        return

    max_scene_count = max(len(original_codes), len(variant_codes))
    for idx in range(max_scene_count):
        scene = scenes[idx] if idx < len(scenes) else None
        shape, dtype, plane_count = scene_metadata(scene)
        scene_idx = scene.scene_idx if scene is not None else idx
        if idx >= len(original_codes) or idx >= len(variant_codes):
            yield converter_error_row(
                sample=sample,
                scene_idx=scene_idx,
                converter=converter,
                data_hamming_threshold=data_hamming_threshold,
                shape=shape,
                dtype=dtype,
                plane_count=plane_count,
                original_path=source_path,
                variant_path=path,
                error=(
                    f"scene count mismatch: original has {len(original_codes)} "
                    f"scene code(s), converted has {len(variant_codes)}"
                ),
                status="mismatch",
            )
            continue
        comparison = compare_entries(
            original_codes[idx],
            variant_codes[idx],
            data_hamming_threshold=data_hamming_threshold,
        )
        yield comparison_row(
            sample=sample,
            scene_idx=scene_idx,
            converter=converter,
            comparison=comparison,
            data_hamming_threshold=data_hamming_threshold,
            shape=shape,
            dtype=dtype,
            plane_count=plane_count,
            original_path=source_path,
            variant_path=path,
        )


def evaluate_sample(
    sample: Sample,
    source_path: Path,
    work_dir: Path,
    *,
    converters: Sequence[Converter],
    tools: dict[str, ToolState],
    external_timeout: int,
    data_hamming_threshold: int,
    max_scene_bytes: int,
) -> Iterator[MatchRow]:
    original_codes = code_for_path(source_path, source_type="bioio")
    scenes = group_by_scene(iter_planes_bioio(source_path))
    if len(original_codes) != len(scenes):
        raise ValueError(
            f"{sample.sample_id}: {len(original_codes)} codes but {len(scenes)} scenes"
        )

    for converter in converters:
        if converter.scope != "sample":
            continue
        yield from evaluate_source_file_converter(
            sample,
            source_path,
            scenes,
            original_codes,
            work_dir,
            converter,
            tools=tools,
            external_timeout=external_timeout,
            data_hamming_threshold=data_hamming_threshold,
        )

    scene_converters = [
        converter for converter in converters if converter.scope == "scene"
    ]
    for scene in scenes:
        scene_dir = work_dir / sample.sample_id / f"scene-{scene.scene_idx:03d}"
        array = scene_to_tczyx(scene)
        shape = "x".join(str(part) for part in array.shape)
        dtype = str(array.dtype)
        plane_count = len(scene.planes)
        original_scene_code = original_codes[scene.scene_idx]

        if array.nbytes > max_scene_bytes:
            for converter in scene_converters:
                yield converter_error_row(
                    sample=sample,
                    scene_idx=scene.scene_idx,
                    converter=converter,
                    data_hamming_threshold=data_hamming_threshold,
                    shape=shape,
                    dtype=dtype,
                    plane_count=plane_count,
                    original_path=source_path,
                    error=f"scene has {array.nbytes} bytes, exceeds --max-scene-bytes={max_scene_bytes}",
                    status="skip",
                )
            continue

        for converter in scene_converters:
            path = (
                scene_dir
                / f"{sample.sample_id}.scene-{scene.scene_idx:03d}{converter.suffix}"
            )
            try:
                converted_array = (
                    perturb_array(array, pixels=converter.drift_pixels)
                    if converter.drift_pixels
                    else array
                )
                if converter.writer is None:
                    raise ValueError(f"{converter.variant} has no scene writer")
                converter.writer(converted_array, path)
                variant_codes = code_for_path(path, source_type=converter.source_type)
                if len(variant_codes) != 1:
                    raise ValueError(
                        f"expected one scene code for {converter.variant}, got {len(variant_codes)}"
                    )
                comparison = compare_entries(
                    original_scene_code,
                    variant_codes[0],
                    data_hamming_threshold=data_hamming_threshold,
                )
                yield comparison_row(
                    sample=sample,
                    scene_idx=scene.scene_idx,
                    converter=converter,
                    comparison=comparison,
                    data_hamming_threshold=data_hamming_threshold,
                    shape=shape,
                    dtype=dtype,
                    plane_count=plane_count,
                    original_path=source_path,
                    variant_path=path,
                )
            except Exception as exc:  # keep per-converter failures explicit
                yield converter_error_row(
                    sample=sample,
                    scene_idx=scene.scene_idx,
                    converter=converter,
                    data_hamming_threshold=data_hamming_threshold,
                    shape=shape,
                    dtype=dtype,
                    plane_count=plane_count,
                    original_path=source_path,
                    variant_path=path,
                    error=repr(exc),
                )


def external_tool_versions() -> dict[str, str]:
    """Return versions for optional independent conversion tools."""

    versions: dict[str, str] = {}
    for command in ("bfconvert", "bioformats2raw", "raw2ometiff"):
        executable = shutil.which(command)
        if not executable:
            versions[command] = "not installed"
            continue
        try:
            completed = subprocess.run(  # noqa: S603 - fixed command list from registry
                [executable, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = (completed.stdout or completed.stderr).strip().splitlines()
            versions[command] = output[0] if output else "installed"
        except Exception as exc:  # pragma: no cover - depends on local optional tools
            versions[command] = f"version check failed: {exc!r}"
    return versions


def converter_to_manifest(converter: Converter) -> dict[str, object]:
    """Return a JSON-serializable converter manifest entry."""

    return {
        "variant": converter.variant,
        "tool": converter.tool,
        "target_format": converter.target_format,
        "drift_pixels": converter.drift_pixels,
        "source_type": converter.source_type,
        "suffix": converter.suffix,
        "note": converter.note,
        "scope": converter.scope,
        "writer": converter.writer.__name__ if converter.writer else "",
        "external_tool_ids": list(converter.external_tool_ids),
    }


def sample_error_row(
    sample: Sample,
    *,
    data_hamming_threshold: int,
    error: str,
    source_path: Path | None = None,
    status: str = "error",
) -> MatchRow:
    """Return a row for download, source decoding, or original-code failures."""

    return MatchRow(
        sample_id=sample.sample_id,
        scene_idx=-1,
        variant="source_decode",
        tool="BioIO",
        target_format=sample.format,
        drift_pixels=0,
        status=status,
        data_hamming_threshold=data_hamming_threshold,
        original_path=str(source_path or ""),
        error=error,
    )


def write_rows(rows: Sequence[MatchRow], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows_json = [asdict(row) for row in rows]
    (results_dir / "results.json").write_text(json.dumps(rows_json, indent=2) + "\n")

    with (results_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(asdict(MatchRow("", -1, "", "", "", 0, "")).keys())
        )
        writer.writeheader()
        writer.writerows(rows_json)


def summarize_rows(rows: Sequence[MatchRow], results_dir: Path) -> dict:
    status_counts = Counter(row.status for row in rows)
    exact_rows = [row for row in rows if row.drift_pixels == 0]
    drift_rows = [row for row in rows if row.drift_pixels > 0]
    summary = {
        "rows": len(rows),
        "samples_with_rows": len({row.sample_id for row in rows}),
        "exact_conversion_rows": len(exact_rows),
        "drift_conversion_rows": len(drift_rows),
        "exact_matches": status_counts.get("exact_match", 0),
        "near_matches": status_counts.get("near_match", 0),
        "mismatches": status_counts.get("mismatch", 0),
        "skips": status_counts.get("skip", 0),
        "errors": status_counts.get("error", 0),
        "data_hamming_threshold": rows[0].data_hamming_threshold
        if rows
        else DEFAULT_DATA_HAMMING_THRESHOLD,
        "data_bits": ISCC_UNIT_BITS,
        "results_dir": str(results_dir),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def format_median(values: Sequence[int]) -> str:
    """Return an exact median string without hiding .5 values."""

    if not values:
        return "n/a"
    median = float(np.median(values))
    return str(int(median)) if median.is_integer() else f"{median:.1f}"


def summarize_by_variant(rows: Sequence[MatchRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[MatchRow]] = defaultdict(list)
    for row in rows:
        grouped[row.variant].append(row)

    table = []
    for variant in sorted(grouped):
        variant_rows = grouped[variant]
        distances = [row.data_hamming for row in variant_rows if row.data_hamming >= 0]
        instance_distances = [
            row.instance_hamming for row in variant_rows if row.instance_hamming >= 0
        ]
        statuses = Counter(row.status for row in variant_rows)
        first = variant_rows[0]
        table.append(
            {
                "variant": variant,
                "tool": first.tool,
                "target_format": first.target_format,
                "drift_pixels": first.drift_pixels,
                "rows": len(variant_rows),
                "exact_match": statuses.get("exact_match", 0),
                "near_match": statuses.get("near_match", 0),
                "mismatch": statuses.get("mismatch", 0),
                "skip": statuses.get("skip", 0),
                "error": statuses.get("error", 0),
                "data_hamming_min": min(distances) if distances else "n/a",
                "data_hamming_median": format_median(distances),
                "data_hamming_max": max(distances) if distances else "n/a",
                "instance_hamming_median": format_median(instance_distances),
            }
        )
    return table


def markdown_table(rows: Sequence[dict[str, object]]) -> str:
    headers = [
        "variant",
        "tool",
        "target format",
        "drifted pixels",
        "rows",
        "exact",
        "near",
        "mismatch",
        "skip/error",
        "Data-Code Hamming median (min-max)",
        "Instance-Code Hamming median",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        data_range = f"{row['data_hamming_median']} ({row['data_hamming_min']}-{row['data_hamming_max']})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["tool"]),
                    str(row["target_format"]),
                    str(row["drift_pixels"]),
                    str(row["rows"]),
                    str(row["exact_match"]),
                    str(row["near_match"]),
                    str(row["mismatch"]),
                    f"{row['skip']}/{row['error']}",
                    data_range,
                    str(row["instance_hamming_median"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_paper_table(rows: Sequence[MatchRow], summary: dict, path: Path) -> None:
    table_rows = summarize_by_variant(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "<!-- Generated by experiments/joss_conversion_matching.py; do not edit by hand. -->\n\n"
        "The conversion experiment produced "
        f"{summary['rows']} comparison rows across {summary['samples_with_rows']} public samples. "
        f"The Data-Code near-match threshold was {summary['data_hamming_threshold']} bits "
        f"out of {summary['data_bits']} bits.\n\n" + markdown_table(table_rows) + "\n"
    )
    path.write_text(content)


def write_png_figure(rows: Sequence[MatchRow], summary: dict, path: Path) -> None:
    """Write a compact PNG bar chart for JOSS/Pandoc compatibility."""

    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        ("Exact", summary["exact_matches"], "#2ca25f"),
        ("Near", summary["near_matches"], "#3182bd"),
        ("Mismatch", summary["mismatches"], "#de2d26"),
        ("Skip/error", summary["errors"] + summary["skips"], "#969696"),
    ]
    width, height = 1080, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = "Conversion matching outcomes"
    subtitle = (
        f"Data-Code near-match threshold: {summary['data_hamming_threshold']} / "
        f"{summary['data_bits']} bits"
    )
    draw.text((width // 2, 45), title, fill="#111111", font=font, anchor="mm")
    draw.text((width // 2, 82), subtitle, fill="#333333", font=font, anchor="mm")

    chart_left, chart_top, chart_bottom = 120, 130, 420
    chart_right = width - 80
    draw.line(
        (chart_left, chart_bottom, chart_right, chart_bottom), fill="#333333", width=2
    )
    draw.line(
        (chart_left, chart_top, chart_left, chart_bottom), fill="#333333", width=2
    )
    max_value = max([value for _, value, _ in values] + [1])
    scale = (chart_bottom - chart_top) / max_value
    bar_width = 120
    slot = (chart_right - chart_left) / len(values)
    for idx, (label, value, color) in enumerate(values):
        x_mid = int(chart_left + slot * idx + slot / 2)
        bar_height = int(value * scale)
        x0, x1 = x_mid - bar_width // 2, x_mid + bar_width // 2
        y0, y1 = chart_bottom - bar_height, chart_bottom
        draw.rectangle((x0, y0, x1, y1), fill=color)
        draw.text((x_mid, y0 - 18), str(value), fill="#111111", font=font, anchor="mm")
        draw.text(
            (x_mid, chart_bottom + 32), label, fill="#111111", font=font, anchor="mm"
        )
    draw.text(
        (38, (chart_top + chart_bottom) // 2),
        "rows",
        fill="#111111",
        font=font,
        anchor="mm",
    )
    image.save(path)


def run_experiment(args: argparse.Namespace) -> dict:
    samples = PUBLIC_CORPUS[: args.max_samples] if args.max_samples else PUBLIC_CORPUS
    results_dir = args.results_dir
    conversions_dir = results_dir / "converted"
    rows: list[MatchRow] = []
    manifest = []
    tools = resolve_external_tools(args)
    converters = converter_registry(args.external_tools)

    for sample in samples:
        sample_manifest = {**asdict(sample), "status": "pending", "path": ""}
        source_path: Path | None = None
        try:
            source_path = download_sample(sample, args.cache_dir, offline=args.offline)
            sample_manifest |= {
                "status": "downloaded",
                "path": str(source_path),
                "actual_sha256": sha256(source_path),
            }
            rows.extend(
                evaluate_sample(
                    sample,
                    source_path,
                    conversions_dir,
                    converters=converters,
                    tools=tools,
                    external_timeout=args.external_timeout,
                    data_hamming_threshold=args.data_hamming_threshold,
                    max_scene_bytes=args.max_scene_bytes,
                )
            )
            sample_manifest["status"] = "processed"
        except (
            Exception
        ) as exc:  # continue to make missing optional readers explicit in results
            source_status = "skip" if source_path is not None else "error"
            sample_manifest["status"] = source_status
            sample_manifest["error"] = repr(exc)
            rows.append(
                sample_error_row(
                    sample,
                    data_hamming_threshold=args.data_hamming_threshold,
                    source_path=source_path,
                    error=repr(exc),
                    status=source_status,
                )
            )
        finally:
            manifest.append(sample_manifest)

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "samples": manifest,
        "converters": [converter_to_manifest(converter) for converter in converters],
        "external_tool_mode": args.external_tools,
        "pinned_external_tools": [asdict(spec) for spec in TOOL_ARCHIVES],
        "resolved_external_tools": {
            tool_id: asdict(state) for tool_id, state in tools.items()
        },
        "system_external_tools": external_tool_versions(),
        "excluded_converters": list(EXCLUDED_CONVERTERS),
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n"
    )
    write_rows(rows, results_dir)
    summary = summarize_rows(rows, results_dir)
    if args.paper_table:
        write_paper_table(rows, summary, args.paper_table)
    if args.paper_figure:
        write_png_figure(rows, summary, args.paper_figure)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE, help="download cache directory"
    )
    parser.add_argument(
        "--tool-cache-dir",
        type=Path,
        default=DEFAULT_TOOL_CACHE,
        help="download cache directory for pinned external converter archives",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="results/output directory",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=DEFAULT_PAPER_TABLE,
        help="tracked Markdown table to include from the paper; use empty string to disable",
    )
    parser.add_argument(
        "--paper-figure",
        type=Path,
        default=DEFAULT_PAPER_FIGURE,
        help="tracked PNG figure to include from the paper; use empty string to disable",
    )
    parser.add_argument(
        "--data-hamming-threshold",
        type=int,
        default=DEFAULT_DATA_HAMMING_THRESHOLD,
        help="maximum Data-Code Hamming distance treated as a near match",
    )
    parser.add_argument(
        "--max-scene-bytes",
        type=int,
        default=DEFAULT_MAX_SCENE_BYTES,
        help="skip dense scene materialization above this many bytes",
    )
    parser.add_argument(
        "--external-tools",
        choices=("none", "bftools", "all"),
        default="bftools",
        help=(
            "external converter set to request; default downloads pinned bftools "
            "and runs bfconvert, while 'all' also requests the gated raw pipeline"
        ),
    )
    parser.add_argument(
        "--max-tool-download-bytes",
        type=int,
        default=DEFAULT_TOOL_DOWNLOAD_BYTES,
        help="maximum size accepted for an individual external tool archive",
    )
    parser.add_argument(
        "--allow-large-tool-downloads",
        action="store_true",
        help=(
            "allow the optional bioformats2raw/raw2ometiff archives; combined "
            "download size is over 400 MB"
        ),
    )
    parser.add_argument(
        "--external-timeout",
        type=int,
        default=DEFAULT_EXTERNAL_TIMEOUT,
        help="timeout in seconds for each external converter command",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="limit corpus entries for quick smoke runs; 0 means the full pinned corpus",
    )
    parser.add_argument(
        "--offline", action="store_true", help="use only already cached sample files"
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="print the public corpus manifest and exit",
    )
    return parser


def normalize_optional_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    text = str(path)
    if text == "" or text == ".":
        return None
    return Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.paper_table = normalize_optional_path(args.paper_table)
    args.paper_figure = normalize_optional_path(args.paper_figure)
    if args.list_samples:
        print(json.dumps([asdict(sample) for sample in PUBLIC_CORPUS], indent=2))
        return 0
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2))
    # Mismatch rows are valid measurements for intentionally drifted or tiny
    # scenes; only unexpected processing errors make the experiment fail.
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
