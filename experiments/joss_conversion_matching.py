#!/usr/bin/env python3
"""Reproducible conversion-matching experiment for the JOSS paper.

The experiment downloads a small public bioimage corpus, converts each readable
sample into normalized OME-TIFF and OME-Zarr variants, and verifies whether
``iscc-bio`` produces identical scene-level BioCodes for the original and the
conversions. Matching here is intentionally pixel-canonical: it tests the
IMAGEWALK traversal and canonical byte representation, not raw file identity.

Outputs are JSON and CSV files under ``experiments/results/`` by default. The
sample cache and converted files are ignored by git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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


@dataclass
class MatchRow:
    sample_id: str
    variant: str
    status: str
    scene_count: int = 0
    matching_scenes: int = 0
    original_path: str = ""
    variant_path: str = ""
    error: str = ""


@dataclass
class SceneBundle:
    scene_idx: int
    planes: list[Plane]


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


def write_ome_tiff(array_tczyx: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, array_tczyx, ome=True, metadata={"axes": "TCZYX"})


def write_ome_zarr(array_tczyx: np.ndarray, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output), mode="w", zarr_format=2)
    # ``scale_factors=[]`` keeps the fixture single-scale and fast. Axes are
    # explicit so ``iter_planes_ngff`` can reconstruct T/C/Z/Y/X positions.
    write_image(array_tczyx, root, axes="tczyx", scale_factors=[])


def code_for_path(path: Path, *, source_type: str) -> list[dict]:
    return biocode(path, source_type=source_type, simprints=False, bits=256)


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
        or not all(isinstance(unit, str) for unit in units)
    ):
        raise TypeError(f"unexpected biocode entry shape: {entry!r}")
    return iscc_code, units


def scene_codes_match(left: Sequence[dict], right: Sequence[dict]) -> tuple[int, int]:
    if len(left) != len(right):
        return 0, max(len(left), len(right))

    matching = 0
    total = len(left)
    for left_entry, right_entry in zip(left, right, strict=True):
        if code_identity(left_entry) == code_identity(right_entry):
            matching += 1
    return matching, total


def evaluate_sample(
    sample: Sample, source_path: Path, work_dir: Path
) -> Iterator[MatchRow]:
    original_codes = code_for_path(source_path, source_type="bioio")
    scenes = group_by_scene(iter_planes_bioio(source_path))
    if len(original_codes) != len(scenes):
        raise ValueError(
            f"{sample.sample_id}: {len(original_codes)} codes but {len(scenes)} scenes"
        )

    for scene in scenes:
        scene_dir = work_dir / sample.sample_id / f"scene-{scene.scene_idx:03d}"
        array = scene_to_tczyx(scene)
        variants = {
            "ome_tiff": scene_dir
            / f"{sample.sample_id}.scene-{scene.scene_idx:03d}.ome.tiff",
            "ome_zarr": scene_dir
            / f"{sample.sample_id}.scene-{scene.scene_idx:03d}.zarr",
        }
        write_ome_tiff(array, variants["ome_tiff"])
        write_ome_zarr(array, variants["ome_zarr"])

        original_scene_code = [original_codes[scene.scene_idx]]
        for variant, path in variants.items():
            source_type = "zarr" if variant == "ome_zarr" else "bioio"
            variant_codes = code_for_path(path, source_type=source_type)
            matching, total = scene_codes_match(original_scene_code, variant_codes)
            yield MatchRow(
                sample_id=sample.sample_id,
                variant=variant,
                status="match" if matching == total == 1 else "mismatch",
                scene_count=total,
                matching_scenes=matching,
                original_path=str(source_path),
                variant_path=str(path),
            )


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
            rows.extend(evaluate_sample(sample, source_path, conversions_dir))
        except (
            Exception
        ) as exc:  # continue to make missing optional readers explicit in results
            rows.append(
                MatchRow(
                    sample_id=sample.sample_id,
                    variant="all",
                    status="error",
                    error=repr(exc),
                )
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows_json = [asdict(row) for row in rows]
    (results_dir / "results.json").write_text(json.dumps(rows_json, indent=2) + "\n")

    with (results_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(asdict(MatchRow("", "", "")).keys())
        )
        writer.writeheader()
        writer.writerows(rows_json)

    summary = {
        "samples": len(samples),
        "rows": len(rows),
        "matches": sum(1 for row in rows if row.status == "match"),
        "mismatches": sum(1 for row in rows if row.status == "mismatch"),
        "errors": sum(1 for row in rows if row.status == "error"),
        "results_dir": str(results_dir),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_samples:
        print(json.dumps([asdict(sample) for sample in PUBLIC_CORPUS], indent=2))
        return 0
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["mismatches"] == 0 and summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
