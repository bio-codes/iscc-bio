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
DEFAULT_CACHE = Path("experiments/cache/joss-bioimages")
DEFAULT_RESULTS = Path("experiments/results/joss-conversion-matching")
DEFAULT_PAPER_TABLE = Path("paper/experiment-results.md")
DEFAULT_PAPER_FIGURE = Path("paper/figures/conversion-matching.png")
DEFAULT_DATA_HAMMING_THRESHOLD = 64
DEFAULT_MAX_SCENE_BYTES = 64 * 1024 * 1024
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


PUBLIC_CORPUS: tuple[Sample, ...] = (
    Sample(
        sample_id="ome_tiff_instrument",
        url="https://downloads.openmicroscopy.org/images/OME-TIFF/2010-06/instrument.ome.tiff",
        expected_size=12092,
        sha256="145f6f5cc214bc6ce95afd0cc10b855bfee7b1a761f908fba17eed4fa3610e2d",
        source="Open Microscopy sample images",
        format="OME-TIFF",
        note="Small OME-TIFF with populated instrument metadata.",
    ),
    Sample(
        sample_id="ome_tiff_plate_companion",
        url="https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/plate-companion/well-A2.ome.tiff",
        expected_size=11770,
        sha256="9aabbb58f01a8cf8addbfd3fd0dbb5cc5247a956f8b4cbb56dacaadaaf6b0643",
        source="Open Microscopy sample images",
        format="OME-TIFF",
        note="OME-TIFF plate companion file.",
    ),
    Sample(
        sample_id="metaxpress_thumb_tiff",
        url="https://downloads.openmicroscopy.org/images/MetaXpress/idr0005/Primary_001/2011-04-19-plate-1_A01_s1_thumb_%5BAEF33B1F-1D43-4BEA-B1A6-667C966E2729%5D.tif",
        expected_size=10987,
        sha256="33870c2553e37d0b12c50ae4e8d0567a26ceb6f8e6775944dc65c6762f5b29b7",
        source="Open Microscopy MetaXpress sample",
        format="TIFF",
        note="Molecular Devices MetaXpress thumbnail TIFF.",
    ),
    Sample(
        sample_id="incell3000_bbbc013_tiff",
        url="https://downloads.openmicroscopy.org/images/InCell3000/BBBC013/BBBC013_v1_images_frm/20041103%201049_01_REF-1049-03%20-%20EvoTec_0_A1_0.tiff",
        expected_size=819398,
        sha256="31dfb71c73ae675932935b2e58f8541d96cba75a79aa0f28b2a15ba49f32ff98",
        source="Open Microscopy / BBBC013 InCell3000 sample",
        format="TIFF",
        note="High-content-screening fluorescence TIFF.",
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


@dataclass(frozen=True)
class Converter:
    """A concrete conversion condition used by the experiment."""

    variant: str
    tool: str
    target_format: str
    drift_pixels: int
    writer: Callable[[np.ndarray, Path], None]
    source_type: str
    suffix: str
    note: str


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


def converter_registry() -> tuple[Converter, ...]:
    """Return the built-in conversion conditions.

    The first three are exact round trips through two Python writer stacks and
    two target bioimage storage formats. The drift variants deliberately apply a
    one-pixel perturbation before writing the same target formats to validate
    Data-Code near matching when Instance-Code equality fails.
    """

    return (
        Converter(
            variant="ome_tiff_tifffile",
            tool="tifffile",
            target_format="OME-TIFF",
            drift_pixels=0,
            writer=write_ome_tiff,
            source_type="bioio",
            suffix=".ome.tiff",
            note="OME-TIFF written by tifffile with explicit TCZYX axes.",
        ),
        Converter(
            variant="ome_tiff_tifffile_deflate",
            tool="tifffile(deflate)",
            target_format="OME-TIFF",
            drift_pixels=0,
            writer=write_ome_tiff_deflate,
            source_type="bioio",
            suffix=".deflate.ome.tiff",
            note="Compressed OME-TIFF written by tifffile/DEFLATE.",
        ),
        Converter(
            variant="ome_zarr_ome_zarr_py",
            tool="ome-zarr-py",
            target_format="OME-Zarr",
            drift_pixels=0,
            writer=write_ome_zarr,
            source_type="zarr",
            suffix=".zarr",
            note="Single-scale OME-NGFF/Zarr written by ome-zarr-py.",
        ),
        Converter(
            variant="ome_tiff_tifffile_one_pixel_drift",
            tool="tifffile+synthetic-drift",
            target_format="OME-TIFF",
            drift_pixels=1,
            writer=write_ome_tiff,
            source_type="bioio",
            suffix=".drift.ome.tiff",
            note="OME-TIFF after deterministic one-pixel +1 perturbation.",
        ),
        Converter(
            variant="ome_zarr_ome_zarr_py_one_pixel_drift",
            tool="ome-zarr-py+synthetic-drift",
            target_format="OME-Zarr",
            drift_pixels=1,
            writer=write_ome_zarr,
            source_type="zarr",
            suffix=".drift.zarr",
            note="OME-Zarr after deterministic one-pixel +1 perturbation.",
        ),
    )


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


def evaluate_sample(
    sample: Sample,
    source_path: Path,
    work_dir: Path,
    *,
    data_hamming_threshold: int,
    max_scene_bytes: int,
) -> Iterator[MatchRow]:
    original_codes = code_for_path(source_path, source_type="bioio")
    scenes = group_by_scene(iter_planes_bioio(source_path))
    if len(original_codes) != len(scenes):
        raise ValueError(
            f"{sample.sample_id}: {len(original_codes)} codes but {len(scenes)} scenes"
        )

    converters = converter_registry()
    for scene in scenes:
        scene_dir = work_dir / sample.sample_id / f"scene-{scene.scene_idx:03d}"
        array = scene_to_tczyx(scene)
        shape = "x".join(str(part) for part in array.shape)
        dtype = str(array.dtype)
        plane_count = len(scene.planes)
        original_scene_code = original_codes[scene.scene_idx]

        if array.nbytes > max_scene_bytes:
            for converter in converters:
                yield MatchRow(
                    sample_id=sample.sample_id,
                    scene_idx=scene.scene_idx,
                    variant=converter.variant,
                    tool=converter.tool,
                    target_format=converter.target_format,
                    drift_pixels=converter.drift_pixels,
                    status="skip",
                    data_hamming_threshold=data_hamming_threshold,
                    shape_tczyx=shape,
                    dtype=dtype,
                    plane_count=plane_count,
                    original_path=str(source_path),
                    error=f"scene has {array.nbytes} bytes, exceeds --max-scene-bytes={max_scene_bytes}",
                )
            continue

        for converter in converters:
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
                yield MatchRow(
                    sample_id=sample.sample_id,
                    scene_idx=scene.scene_idx,
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
                    original_path=str(source_path),
                    variant_path=str(path),
                )
            except Exception as exc:  # keep per-converter failures explicit
                yield MatchRow(
                    sample_id=sample.sample_id,
                    scene_idx=scene.scene_idx,
                    variant=converter.variant,
                    tool=converter.tool,
                    target_format=converter.target_format,
                    drift_pixels=converter.drift_pixels,
                    status="error",
                    data_hamming_threshold=data_hamming_threshold,
                    shape_tczyx=shape,
                    dtype=dtype,
                    plane_count=plane_count,
                    original_path=str(source_path),
                    variant_path=str(path),
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

    for sample in samples:
        try:
            source_path = download_sample(sample, args.cache_dir, offline=args.offline)
            manifest.append(
                {
                    **asdict(sample),
                    "path": str(source_path),
                    "sha256": sha256(source_path),
                }
            )
            rows.extend(
                evaluate_sample(
                    sample,
                    source_path,
                    conversions_dir,
                    data_hamming_threshold=args.data_hamming_threshold,
                    max_scene_bytes=args.max_scene_bytes,
                )
            )
        except (
            Exception
        ) as exc:  # continue to make missing optional readers explicit in results
            rows.append(
                MatchRow(
                    sample_id=sample.sample_id,
                    scene_idx=-1,
                    variant="all",
                    tool="all",
                    target_format="all",
                    drift_pixels=0,
                    status="error",
                    data_hamming_threshold=args.data_hamming_threshold,
                    error=repr(exc),
                )
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "samples": manifest,
        "converters": [
            asdict(converter) | {"writer": converter.writer.__name__}
            for converter in converter_registry()
        ],
        "optional_external_tools": external_tool_versions(),
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
        "--max-samples",
        type=int,
        default=0,
        help="limit corpus entries for quick smoke runs",
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
    if path is None or str(path) == "":
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
