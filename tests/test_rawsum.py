"""Tests for raw-byte ISCC-SUM over files and directory trees (rawsum)."""

import os

import iscc_lib

from iscc_bio.rawsum import raw_sum
from iscc_bio.treewalk import treewalk_iscc


def test_single_file_matches_gen_sum_code(tmp_path):
    """raw_sum over a single file equals the one-shot iscc_lib.gen_sum_code_v0."""
    f = tmp_path / "data.bin"
    f.write_bytes(os.urandom(3_000_000))  # spans multiple 2 MiB chunks
    got = raw_sum(f)
    ref = iscc_lib.gen_sum_code_v0(str(f), bits=256, wide=True, add_units=True)
    assert got["iscc_code"] == ref["iscc"]
    assert got["units"] == ref["units"]
    assert got["filesize"] == ref["filesize"]


def test_result_shape(tmp_path):
    """raw_sum returns iscc_code, two units, datahash, and filesize."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    r = raw_sum(f)
    assert r["iscc_code"].startswith("ISCC:")
    assert isinstance(r["units"], list) and len(r["units"]) == 2
    assert r["filesize"] == 11
    assert r["datahash"].startswith("1e20")  # blake3 multihash prefix


def test_tree_sum_matches_manual_treewalk(tmp_path):
    """Directory SUM equals feeding treewalk_iscc-ordered bytes into a SumHasher."""
    (tmp_path / "b.bin").write_bytes(b"BBB")
    (tmp_path / "a.bin").write_bytes(b"AAA")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.bin").write_bytes(b"CCC")

    got = raw_sum(tmp_path)

    h = iscc_lib.SumHasher()
    for fp in treewalk_iscc(tmp_path):
        h.update(fp.read_bytes())
    expected = h.finalize(bits=256, wide=True, add_units=True)
    assert got["iscc_code"] == expected["iscc"]


def test_tree_sum_deterministic_regardless_of_creation_order(tmp_path):
    """Trees with identical files created in different order hash identically."""
    d1 = tmp_path / "d1"
    d1.mkdir()
    (d1 / "a.bin").write_bytes(b"AAA")
    (d1 / "z.bin").write_bytes(b"ZZZ")
    (d1 / "m.bin").write_bytes(b"MMM")

    d2 = tmp_path / "d2"
    d2.mkdir()
    (d2 / "z.bin").write_bytes(b"ZZZ")
    (d2 / "m.bin").write_bytes(b"MMM")
    (d2 / "a.bin").write_bytes(b"AAA")

    assert raw_sum(d1)["iscc_code"] == raw_sum(d2)["iscc_code"]


def test_iscc_sidecar_does_not_affect_tree_sum(tmp_path):
    """Adding an ISCC .iscc.json sidecar leaves the tree SUM unchanged."""
    (tmp_path / "data.txt").write_bytes(b"payload")
    before = raw_sum(tmp_path)
    (tmp_path / "data.txt.iscc.json").write_text('{"iscc": "ISCC:..."}')
    after = raw_sum(tmp_path)
    assert before["iscc_code"] == after["iscc_code"]
    assert before["filesize"] == after["filesize"]


def test_tree_sum_differs_on_content_change(tmp_path):
    """Different file content yields a different tree code."""
    d1 = tmp_path / "d1"
    d1.mkdir()
    (d1 / "a.bin").write_bytes(b"AAA")
    d2 = tmp_path / "d2"
    d2.mkdir()
    (d2 / "a.bin").write_bytes(b"BBB")
    assert raw_sum(d1)["iscc_code"] != raw_sum(d2)["iscc_code"]
