"""Tests for the JOSS conversion-matching experiment helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from iscc_bio.biocode import generate_biocode
from iscc_bio.imagewalk import iter_planes_bioio, iter_planes_ngff
from iscc_bio.imagewalk.common import Plane


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "experiments" / "joss_conversion_matching.py"
)
SPEC = importlib.util.spec_from_file_location("joss_conversion_matching", SCRIPT_PATH)
assert SPEC and SPEC.loader
joss = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = joss
SPEC.loader.exec_module(joss)


def make_planes() -> list[Plane]:
    """Create a tiny deterministic multi-Z, multi-channel scene."""

    planes = []
    for z in range(3):
        for c in range(2):
            base = z * 100 + c * 10
            data = (np.arange(8 * 9, dtype=np.uint16).reshape(8, 9) + base).astype(
                np.uint16
            )
            planes.append(
                Plane(xy_array=data, scene_idx=0, z_depth=z, c_channel=c, t_time=0)
            )
    return planes


def test_scene_to_tczyx_preserves_imagewalk_order():
    scene = joss.SceneBundle(scene_idx=0, planes=make_planes())
    array = joss.scene_to_tczyx(scene)

    assert array.shape == (1, 2, 3, 8, 9)
    assert int(array[0, 0, 0, 0, 0]) == 0
    assert int(array[0, 1, 0, 0, 0]) == 10
    assert int(array[0, 0, 1, 0, 0]) == 100
    assert int(array[0, 1, 2, 0, 0]) == 210


def test_scene_codes_match_fails_on_missing_identity_keys():
    with pytest.raises(KeyError):
        joss.scene_codes_match(
            [{"iscc": "ISCC:missing-schema"}], [{"iscc": "ISCC:missing-schema"}]
        )


def test_scene_codes_match_detects_mismatch():
    left = [{"iscc_code": "ISCC:AAA", "units": ["ISCC:DATA", "ISCC:INST"]}]
    right = [{"iscc_code": "ISCC:BBB", "units": ["ISCC:DATA", "ISCC:INST"]}]

    assert joss.scene_codes_match(left, right) == (0, 1)


def test_scene_codes_match_requires_same_number_of_entries():
    entry = {"iscc_code": "ISCC:AAA", "units": ["ISCC:DATA", "ISCC:INST"]}

    assert joss.scene_codes_match([entry], [entry, entry]) == (0, 2)


def test_one_pixel_drift_changes_instance_but_keeps_data_near_match():
    original_plane = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    drifted_plane = original_plane.copy()
    drifted_plane[0, 0] += 1
    original = generate_biocode(
        iter([Plane(original_plane, scene_idx=0, z_depth=0, c_channel=0, t_time=0)])
    )[0]
    drifted = generate_biocode(
        iter([Plane(drifted_plane, scene_idx=0, z_depth=0, c_channel=0, t_time=0)])
    )[0]

    comparison = joss.compare_entries(original, drifted, data_hamming_threshold=64)

    assert comparison.status == "near_match"
    assert comparison.data_near_match is True
    assert comparison.data_code_equal is False
    assert comparison.instance_code_equal is False
    assert 0 < comparison.data_hamming <= 64
    assert comparison.instance_hamming > comparison.data_hamming


def test_generated_paper_table_mentions_threshold(tmp_path):
    original_plane = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    drifted_plane = original_plane.copy()
    drifted_plane[0, 0] += 1
    original = generate_biocode(
        iter([Plane(original_plane, scene_idx=0, z_depth=0, c_channel=0, t_time=0)])
    )[0]
    drifted = generate_biocode(
        iter([Plane(drifted_plane, scene_idx=0, z_depth=0, c_channel=0, t_time=0)])
    )[0]
    comparison = joss.compare_entries(original, drifted, data_hamming_threshold=64)
    row = joss.MatchRow(
        sample_id="synthetic",
        scene_idx=0,
        variant="drift",
        tool="synthetic",
        target_format="OME-TIFF",
        drift_pixels=1,
        status=comparison.status,
        data_near_match=comparison.data_near_match,
        data_code_equal=comparison.data_code_equal,
        instance_code_equal=comparison.instance_code_equal,
        data_hamming=comparison.data_hamming,
        instance_hamming=comparison.instance_hamming,
        data_hamming_threshold=64,
    )
    summary = joss.summarize_rows([row], tmp_path)
    table = tmp_path / "table.md"

    joss.write_paper_table([row], summary, table)

    text = table.read_text()
    assert "Data-Code near-match threshold was 64 bits" in text
    assert "drift" in text
    assert "near" in text.lower()


def test_synthetic_ome_tiff_roundtrip_matches_biocode(tmp_path):
    original = generate_biocode(iter(make_planes()))
    scene = joss.SceneBundle(scene_idx=0, planes=make_planes())
    output = tmp_path / "synthetic.ome.tiff"

    joss.write_ome_tiff(joss.scene_to_tczyx(scene), output)
    converted = generate_biocode(iter_planes_bioio(output))

    assert converted == original


def test_synthetic_ome_zarr_roundtrip_matches_biocode(tmp_path):
    original = generate_biocode(iter(make_planes()))
    scene = joss.SceneBundle(scene_idx=0, planes=make_planes())
    output = tmp_path / "synthetic.zarr"

    joss.write_ome_zarr(joss.scene_to_tczyx(scene), output)
    converted = generate_biocode(iter_planes_ngff(output))

    assert converted == original
