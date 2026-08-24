"""Offline tests for the public bioimage test-data acquisition script.

These tests exercise the pinned corpus, filename derivation, digest
verification, and offline caching behavior without touching the network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "acquire_testdata.py"
SPEC = importlib.util.spec_from_file_location("acquire_testdata", SCRIPT_PATH)
assert SPEC and SPEC.loader
acq = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acq
SPEC.loader.exec_module(acq)


def make_sample(content: bytes, **overrides) -> "acq.Sample":
    """Build a corpus entry whose size and digest match ``content``."""

    fields = dict(
        sample_id="fixture_sample",
        url="https://example.org/fixture.tif",
        expected_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        source="unit test",
        format="TIFF",
        note="local fixture",
    )
    fields.update(overrides)
    return acq.Sample(**fields)


class FakeResponse:
    """Stand-in for ``urllib.request.urlopen``: callable, context manager, reader."""

    def __init__(self, payload: bytes, content_length: int | None = None):
        self._data = io.BytesIO(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __call__(self, url, timeout=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size: int) -> bytes:
        return self._data.read(size)


def test_public_corpus_covers_pinned_broad_formats():
    by_id = {sample.sample_id: sample for sample in acq.PUBLIC_CORPUS}
    formats = {sample.format for sample in acq.PUBLIC_CORPUS}

    assert {"OME-TIFF", "TIFF", "OIR", "LIF", "ND2", "CZI"} <= formats
    assert by_id["ome_tiff_multi_channel_4d"].expected_size == 7_889_665
    assert (
        by_id["ome_tiff_multi_channel_4d"].sha256
        == "23ec5b84154850360800b299e6c088b8f60c5e81b6c990ac1e9b15496fa9549d"
    )
    assert by_id["nikon_nd2_bf007"].expected_size == 270_336
    assert "S-BIAD1" in by_id["bia_sbiad1_rnafish_tiff"].url


def test_corpus_entries_are_pinned_and_unique():
    ids = [sample.sample_id for sample in acq.PUBLIC_CORPUS]
    urls = [sample.url for sample in acq.PUBLIC_CORPUS]

    assert len(ids) == len(set(ids))
    assert len(urls) == len(set(urls))
    for sample in acq.PUBLIC_CORPUS:
        assert sample.expected_size > 0
        assert len(sample.sha256) == 64
        assert sample.expected_size <= acq.MAX_DOWNLOAD_BYTES


def test_sample_filename_handles_double_and_missing_suffix():
    ome = next(s for s in acq.PUBLIC_CORPUS if s.format == "OME-TIFF")
    nd2 = next(s for s in acq.PUBLIC_CORPUS if s.format == "ND2")

    assert acq.sample_filename(ome) == "ome_tiff_multi_channel_4d.ome.tiff"
    assert acq.sample_filename(nd2) == "nikon_nd2_bf007.nd2"

    no_suffix = acq.Sample(
        sample_id="no_suffix",
        url="https://example.org/download?id=42",
        expected_size=1,
        sha256="0" * 64,
        source="unit test",
        format="TIFF",
        note="",
    )
    assert acq.sample_filename(no_suffix) == "no_suffix.bin"


def test_verify_sample_file_accepts_matching_file(tmp_path):
    content = b"hello bioimage corpus"
    sample = make_sample(content)
    path = tmp_path / "fixture.tif"
    path.write_bytes(content)

    acq.verify_sample_file(sample, path)  # must not raise


def test_verify_sample_file_rejects_size_mismatch(tmp_path):
    sample = make_sample(b"original content")
    path = tmp_path / "fixture.tif"
    path.write_bytes(b"shorter")

    with pytest.raises(ValueError, match="bytes"):
        acq.verify_sample_file(sample, path)


def test_verify_sample_file_rejects_digest_mismatch(tmp_path):
    content = b"same length, different bytes"
    sample = make_sample(content)
    path = tmp_path / "fixture.tif"
    path.write_bytes(b"X" * len(content))

    with pytest.raises(ValueError, match="sha256"):
        acq.verify_sample_file(sample, path)


def test_download_sample_returns_verified_cache_offline(tmp_path):
    content = b"cached pixel bytes"
    sample = make_sample(content)
    (tmp_path / acq.sample_filename(sample)).write_bytes(content)

    path = acq.download_sample(sample, tmp_path, offline=True)

    assert path == tmp_path / acq.sample_filename(sample)


def test_download_sample_offline_without_cache_raises(tmp_path):
    sample = make_sample(b"never downloaded")

    with pytest.raises(FileNotFoundError):
        acq.download_sample(sample, tmp_path, offline=True)


def test_download_sample_streams_verifies_and_promotes(tmp_path, monkeypatch):
    content = b"streamed pixel bytes"
    sample = make_sample(content)
    monkeypatch.setattr("urllib.request.urlopen", FakeResponse(content, len(content)))

    path = acq.download_sample(sample, tmp_path, offline=False)

    assert path == tmp_path / acq.sample_filename(sample)
    assert path.read_bytes() == content
    assert not list(tmp_path.glob("*.tmp"))


def test_download_sample_digest_mismatch_leaves_no_file(tmp_path, monkeypatch):
    content = b"expected corpus bytes"
    sample = make_sample(content)
    monkeypatch.setattr("urllib.request.urlopen", FakeResponse(b"X" * len(content)))

    with pytest.raises(ValueError, match="sha256"):
        acq.download_sample(sample, tmp_path, offline=False)

    assert not list(tmp_path.iterdir())


def test_download_sample_rejects_reported_oversize(tmp_path, monkeypatch):
    sample = make_sample(b"tiny")
    monkeypatch.setattr(acq, "MAX_DOWNLOAD_BYTES", 8)
    monkeypatch.setattr("urllib.request.urlopen", FakeResponse(b"irrelevant", 100))

    with pytest.raises(ValueError, match="server reports"):
        acq.download_sample(sample, tmp_path, offline=False)

    assert not list(tmp_path.iterdir())


def test_download_sample_enforces_size_cap_during_stream(tmp_path, monkeypatch):
    sample = make_sample(b"tiny")
    monkeypatch.setattr(acq, "MAX_DOWNLOAD_BYTES", 8)
    monkeypatch.setattr(
        "urllib.request.urlopen", FakeResponse(b"more than eight bytes")
    )

    with pytest.raises(ValueError, match="downloaded over"):
        acq.download_sample(sample, tmp_path, offline=False)

    assert not list(tmp_path.iterdir())


def test_select_samples_defaults_to_full_corpus():
    assert acq.select_samples(acq.PUBLIC_CORPUS, ()) == list(acq.PUBLIC_CORPUS)


def test_select_samples_filters_by_id_in_corpus_order():
    selected = acq.select_samples(
        acq.PUBLIC_CORPUS, ["nikon_nd2_bf007", "tiff_condensation_c4"]
    )

    assert [sample.sample_id for sample in selected] == [
        "nikon_nd2_bf007",
        "tiff_condensation_c4",
    ]


def test_select_samples_rejects_unknown_id():
    with pytest.raises(SystemExit, match="unknown sample id"):
        acq.select_samples(acq.PUBLIC_CORPUS, ["does_not_exist"])


def test_acquire_sample_reports_cached_status(tmp_path):
    content = b"already here"
    sample = make_sample(content)
    (tmp_path / acq.sample_filename(sample)).write_bytes(content)

    result = acq.acquire_sample(sample, tmp_path, offline=True)

    assert result.status == "cached"
    assert result.size == len(content)
    assert result.error == ""


def test_acquire_sample_reports_error_for_missing_offline(tmp_path):
    sample = make_sample(b"missing")

    result = acq.acquire_sample(sample, tmp_path, offline=True)

    assert result.status == "error"
    assert result.size == 0
    assert "FileNotFoundError" in result.error


def test_main_list_prints_full_corpus(capsys):
    exit_code = acq.main(["--list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ome_tiff_multi_channel_4d" in captured.out
    assert "leica_lif_pr2729" in captured.out
