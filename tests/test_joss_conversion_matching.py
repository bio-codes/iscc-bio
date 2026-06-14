"""Tests for the JOSS conversion-matching experiment helpers."""

from __future__ import annotations

import argparse
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


def test_public_corpus_contains_pinned_broad_formats():
    by_format = {sample.format: sample for sample in joss.PUBLIC_CORPUS}

    assert {"OME-TIFF", "TIFF", "OIR", "LIF", "ND2", "CZI"} <= set(by_format)
    assert by_format["OME-TIFF"].expected_size == 7_889_665
    assert (
        by_format["OME-TIFF"].sha256
        == "23ec5b84154850360800b299e6c088b8f60c5e81b6c990ac1e9b15496fa9549d"
    )
    assert by_format["ND2"].expected_size == 270_336
    assert (
        by_format["CZI"].sha256
        == "44593e6210f2f9066f8608c53c31806f4d173d1d26bb3bb5c32b182fb0c0a43e"
    )


def test_pinned_external_tool_manifest_and_gating(tmp_path):
    specs = {spec.tool_id: spec for spec in joss.TOOL_ARCHIVES}

    assert specs["bftools"].version == "8.5.0"
    assert specs["bftools"].expected_size == 51_360_836
    assert (
        specs["bftools"].sha256
        == "07a3bb1d3de84da3a709655a1008cb2d9b19becc5bad4ae4112633aec9380478"
    )
    assert specs["bioformats2raw"].large_download is True
    assert specs["raw2ometiff"].large_download is True

    args = argparse.Namespace(
        external_tools="all",
        allow_large_tool_downloads=False,
        max_tool_download_bytes=joss.DEFAULT_TOOL_DOWNLOAD_BYTES,
        tool_cache_dir=tmp_path,
        offline=True,
    )
    states = joss.resolve_external_tools(args)

    assert states["bioformats2raw"].status == "gated"
    assert states["raw2ometiff"].status == "gated"


def test_converter_registry_modes_include_real_and_gated_converters():
    no_external = {converter.variant for converter in joss.converter_registry("none")}
    default = {converter.variant for converter in joss.converter_registry("bftools")}
    all_converters = {converter.variant for converter in joss.converter_registry("all")}

    assert "ome_tiff_bfconvert" not in no_external
    assert "ome_tiff_bfconvert" in default
    assert "ome_tiff_bioformats2raw_raw2ometiff" not in default
    assert "ome_tiff_bioformats2raw_raw2ometiff" in all_converters


def test_parser_defaults_request_small_bftools_run():
    args = joss.build_parser().parse_args([])

    assert args.max_samples == 0
    assert args.external_tools == "bftools"
    assert args.allow_large_tool_downloads is False


def test_empty_optional_artifact_paths_disable_writes():
    assert joss.normalize_optional_path("") is None
    assert joss.normalize_optional_path(Path(".")) is None
    assert joss.normalize_optional_path(Path("paper/experiment-results.md")) == Path(
        "paper/experiment-results.md"
    )


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


def test_source_file_bfconvert_path_compares_converted_scenes(tmp_path):
    scene = joss.SceneBundle(scene_idx=0, planes=make_planes())
    source = tmp_path / "source.ome.tiff"
    joss.write_ome_tiff(joss.scene_to_tczyx(scene), source)
    fake_bfconvert = tmp_path / "bfconvert"
    fake_bfconvert.write_text('#!/bin/sh\ncp "$1" "$2"\n')
    fake_bfconvert.chmod(0o755)
    tools = {
        "bftools": joss.ToolState(
            tool_id="bftools",
            label="Bio-Formats command-line tools",
            version="8.5.0",
            status="ready",
            executable=str(fake_bfconvert),
        )
    }
    converter = next(
        converter
        for converter in joss.converter_registry("bftools")
        if converter.variant == "ome_tiff_bfconvert"
    )
    sample = joss.Sample(
        sample_id="synthetic_ome_tiff",
        url="file://synthetic",
        expected_size=source.stat().st_size,
        sha256=joss.sha256(source),
        source="synthetic",
        format="OME-TIFF",
        note="test fixture",
    )
    original_codes = joss.code_for_path(source, source_type="bioio")
    scenes = joss.group_by_scene(iter_planes_bioio(source))

    rows = list(
        joss.evaluate_source_file_converter(
            sample,
            source,
            scenes,
            original_codes,
            tmp_path / "converted",
            converter,
            tools=tools,
            external_timeout=15,
            data_hamming_threshold=64,
        )
    )

    assert len(rows) == 1
    assert rows[0].variant == "ome_tiff_bfconvert"
    assert rows[0].status == "exact_match"
    assert rows[0].data_hamming == 0
    assert Path(rows[0].variant_path).exists()
