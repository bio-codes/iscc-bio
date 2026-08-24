#!/usr/bin/env python3
"""Acquire a pinned corpus of diverse public bioimage samples for testing.

The corpus spans the proprietary microscopy formats and open container formats
that ``iscc-bio`` must read identically (OME-TIFF, plain TIFF, Zeiss CZI, Nikon
ND2, Olympus OIR, Leica LIF). Every entry pins an exact URL, byte size, and
SHA-256 digest so a download is reproducible and tamper-evident: a cached file
is re-verified on every run and a size or digest mismatch fails loudly.

Samples download into a git-ignored project subfolder (``testdata/`` by default)
so acquired pixel data never enters version control. The script depends only on
the Python standard library, so it runs without the heavy bioimage reader stack.

Examples::

    uv run python scripts/acquire_testdata.py             # download the full corpus
    uv run python scripts/acquire_testdata.py --list      # print the manifest as JSON
    uv run python scripts/acquire_testdata.py --only nikon_nd2_bf007
    uv run python scripts/acquire_testdata.py --offline   # verify cached files only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "testdata"
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DOWNLOAD_TIMEOUT = 60
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Sample:
    """Pinned public bioimage corpus entry."""

    sample_id: str
    url: str
    expected_size: int
    sha256: str
    source: str
    format: str
    note: str


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of acquiring one corpus entry."""

    sample_id: str
    format: str
    status: str
    path: str
    size: int
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
        sample_id="bia_sbiad1_rnafish_tiff",
        url="https://www.ebi.ac.uk/biostudies/files/S-BIAD1/20181016-ftp/Exp2_rep3/%232_Analyzed_images/SD_mRNA_Exp2_rep3_0min_im5_TMRmaxF.tif",
        expected_size=131796,
        sha256="f5ff8a20890a89f3767a788e29007dc7e15e62e80120fe874f8ea3fc187fad1c",
        source="BioImage Archive S-BIAD1 RNA-FISH sample",
        format="TIFF",
        note="Small BioImage Archive analyzed RNA-FISH TIFF fixture.",
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


def sample_filename(sample: Sample) -> str:
    """Return a deterministic local filename for a sample URL."""

    suffix = Path(urllib.request.url2pathname(sample.url.rsplit("/", 1)[-1])).suffix
    if sample.url.endswith(".ome.tiff"):
        suffix = ".ome.tiff"
    if not suffix:
        suffix = ".bin"
    return f"{sample.sample_id}{suffix}"


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in fixed-size chunks."""

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_sample_file(sample: Sample, path: Path) -> None:
    """Verify a cached sample against its pinned size and SHA-256 digest."""

    actual_size = path.stat().st_size
    if actual_size != sample.expected_size:
        raise ValueError(
            f"{sample.sample_id}: expected {sample.expected_size} bytes "
            f"but found {actual_size}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != sample.sha256:
        raise ValueError(
            f"{sample.sample_id}: expected sha256 {sample.sha256} "
            f"but found {actual_sha256}"
        )


def download_sample(sample: Sample, data_dir: Path, *, offline: bool) -> Path:
    """Download ``sample`` into ``data_dir`` unless it is already cached.

    A cached file is re-verified and returned without a network call. A fresh
    download is streamed to a temporary file, size-capped during transfer, and
    only promoted to the final path after passing verification. A failed
    download or verification removes the temporary file and leaves no trace.
    """

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / sample_filename(sample)
    if target.exists():
        verify_sample_file(sample, target)
        return target
    if offline:
        raise FileNotFoundError(f"{target} is not cached and --offline was requested")
    if sample.expected_size > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"refusing to download {sample.sample_id}: expected size "
            f"{sample.expected_size} exceeds {MAX_DOWNLOAD_BYTES}"
        )

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed public sample URLs
            sample.url, timeout=DOWNLOAD_TIMEOUT
        ) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"refusing to download {sample.sample_id}: "
                    f"server reports {length} bytes"
                )
            total = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"refusing to keep {sample.sample_id}: "
                            f"downloaded over {MAX_DOWNLOAD_BYTES} bytes"
                        )
                    out.write(chunk)
        verify_sample_file(sample, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(target)
    return target


def select_samples(corpus: Sequence[Sample], only: Iterable[str]) -> list[Sample]:
    """Return corpus entries matching ``only`` ids, preserving corpus order.

    An empty ``only`` selects the whole corpus. Unknown ids fail loudly so a
    typo never silently downloads nothing.
    """

    wanted = list(only)
    if not wanted:
        return list(corpus)
    by_id = {sample.sample_id: sample for sample in corpus}
    unknown = [sample_id for sample_id in wanted if sample_id not in by_id]
    if unknown:
        known = ", ".join(sample.sample_id for sample in corpus)
        raise SystemExit(
            f"unknown sample id(s): {', '.join(unknown)}\nknown ids: {known}"
        )
    return [by_id[sample_id] for sample_id in wanted]


def acquire_sample(
    sample: Sample, data_dir: Path, *, offline: bool
) -> AcquisitionResult:
    """Acquire one sample, capturing any failure as a result row."""

    target = data_dir / sample_filename(sample)
    was_cached = target.exists()
    try:
        path = download_sample(sample, data_dir, offline=offline)
    except Exception as exc:  # keep per-sample failures explicit and non-fatal
        return AcquisitionResult(
            sample_id=sample.sample_id,
            format=sample.format,
            status="error",
            path=str(target),
            size=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    return AcquisitionResult(
        sample_id=sample.sample_id,
        format=sample.format,
        status="cached" if was_cached else "downloaded",
        path=str(path),
        size=path.stat().st_size,
    )


def acquire_corpus(
    samples: Sequence[Sample], data_dir: Path, *, offline: bool
) -> list[AcquisitionResult]:
    """Acquire every sample in ``samples`` and return one result per entry."""

    return [acquire_sample(sample, data_dir, offline=offline) for sample in samples]


def print_results(results: Sequence[AcquisitionResult], data_dir: Path) -> None:
    """Print a per-sample status line and an aggregate summary."""

    for result in results:
        detail = result.error if result.status == "error" else f"{result.size} bytes"
        print(
            f"  [{result.status:<10}] {result.sample_id} ({result.format}) - {detail}"
        )

    downloaded = sum(result.status == "downloaded" for result in results)
    cached = sum(result.status == "cached" for result in results)
    errors = sum(result.status == "error" for result in results)
    total_bytes = sum(result.size for result in results)
    print(
        f"\n{len(results)} sample(s): {downloaded} downloaded, {cached} cached, "
        f"{errors} error(s); {total_bytes} bytes in {data_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser for the acquisition script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="git-ignored directory for downloaded samples (default: %(default)s)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=(),
        metavar="SAMPLE_ID",
        help="restrict acquisition to specific sample ids (default: full corpus)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify already cached samples without any network access",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the pinned corpus manifest as JSON and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the acquisition CLI, returning a process exit code."""

    args = build_parser().parse_args(argv)
    if args.list:
        print(json.dumps([asdict(sample) for sample in PUBLIC_CORPUS], indent=2))
        return 0

    samples = select_samples(PUBLIC_CORPUS, args.only)
    results = acquire_corpus(samples, args.data_dir, offline=args.offline)
    print_results(results, args.data_dir)
    return 1 if any(result.status == "error" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
