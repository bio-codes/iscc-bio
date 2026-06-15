"""Tests for biocode (scene-level ISCC-SUM) generation."""

import numpy as np
import iscc_lib
from iscc_bio.imagewalk.common import Plane
from iscc_bio.biocode import _selected_mixed_content_code, generate_biocode


def make_plane(scene_idx=0, z=0, c=0, t=0, shape=(64, 64), dtype=np.uint8, value=None):
    """Create a Plane with deterministic or random data."""
    if value is not None:
        data = np.full(shape, value, dtype=dtype)
    else:
        rng = np.random.default_rng(seed=scene_idx * 1000 + z * 100 + c * 10 + t)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            data = rng.integers(info.min, info.max, size=shape, dtype=dtype)
        else:
            data = rng.random(size=shape).astype(dtype)
    return Plane(xy_array=data, scene_idx=scene_idx, z_depth=z, c_channel=c, t_time=t)


def test_single_scene_basic():
    """Single scene produces one IsccEntry."""
    planes = [make_plane(z=z) for z in range(3)]
    results = generate_biocode(iter(planes))
    assert len(results) == 1
    assert "iscc_code" in results[0]
    assert "units" in results[0]
    assert len(results[0]["units"]) == 2
    assert results[0]["iscc_code"].startswith("ISCC:")
    assert results[0]["units"][0].startswith("ISCC:")
    assert results[0]["units"][1].startswith("ISCC:")


def test_multi_scene():
    """Multiple scenes produce one IsccEntry per scene."""
    planes = [
        make_plane(scene_idx=0, z=0),
        make_plane(scene_idx=0, z=1),
        make_plane(scene_idx=1, z=0),
        make_plane(scene_idx=1, z=1),
        make_plane(scene_idx=2, z=0),
    ]
    results = generate_biocode(iter(planes))
    assert len(results) == 3


def test_no_simprints_by_default():
    """Simprints are not included when flag is False."""
    planes = [make_plane(z=z) for z in range(3)]
    results = generate_biocode(iter(planes))
    assert "simprints" not in results[0]


def test_no_content_codes_by_default():
    """Mixed Content-Code sidecars are opt-in."""
    planes = [make_plane(z=z) for z in range(3)]
    results = generate_biocode(iter(planes))

    assert "content_codes" not in results[0]


def test_mixed_content_code_sidecar_selects_three_imagewalk_planes():
    """Opt-in Content-Code-Mixed sidecar selects first/middle/last IMAGEWALK planes."""
    planes = [make_plane(z=z, value=z * 10) for z in range(5)]

    result = generate_biocode(iter(planes), content_codes=True)[0]

    sidecar = result["content_codes"]["CONTENT_MIXED_V0"]
    assert sidecar["iscc"].startswith("ISCC:")
    assert sidecar["offsets"] == [0, 2, 4]
    assert sidecar["plane_count"] == 5
    assert sidecar["selected_count"] == 3
    assert sidecar["bits"] == 256
    assert sidecar["derivation"] == "IMAGEWALK_MIXED_CONTENT_V0"
    assert sidecar["mixed_input_count"] == 3
    assert len(sidecar["image_codes"]) == 3


def test_mixed_content_code_is_standard_content_mixed_unit():
    """The sidecar unit uses ISCC's standard Content-Code-Mixed subtype."""
    planes = [make_plane(z=z) for z in range(3)]
    sidecar = generate_biocode(iter(planes), content_codes=True)[0]["content_codes"][
        "CONTENT_MIXED_V0"
    ]

    main_type, sub_type, version, _length, _body = iscc_lib.iscc_decode(sidecar["iscc"])
    assert main_type.name == "CONTENT"
    assert sub_type.name == "MIXED"
    assert version.name == "V0"


def test_mixed_content_code_single_plane_duplicates_input_for_standard_mixed_code():
    """A single-plane scene still emits a valid standard Mixed-Code by duplicating the plane code."""
    sidecar = generate_biocode(iter([make_plane(value=42)]), content_codes=True)[0][
        "content_codes"
    ]["CONTENT_MIXED_V0"]

    assert sidecar["offsets"] == [0]
    assert sidecar["selected_count"] == 1
    assert sidecar["mixed_input_count"] == 2
    assert len(sidecar["image_codes"]) == 1
    assert sidecar["iscc"].startswith("ISCC:")


def test_mixed_content_code_is_more_stable_than_data_code_for_small_pixel_drift():
    """The perceptual sidecar stays equal for small compression-like pixel drift."""
    rng = np.random.default_rng(1234)
    planes = [
        Plane(
            xy_array=rng.integers(0, 255, size=(96, 96), dtype=np.uint8),
            scene_idx=0,
            z_depth=z,
            c_channel=0,
            t_time=0,
        )
        for z in range(3)
    ]
    drifted = [
        Plane(
            xy_array=np.clip(plane.xy_array.astype(np.int16) + 1, 0, 255).astype(
                np.uint8
            ),
            scene_idx=plane.scene_idx,
            z_depth=plane.z_depth,
            c_channel=plane.c_channel,
            t_time=plane.t_time,
        )
        for plane in planes
    ]

    original = generate_biocode(iter(planes), content_codes=True)[0]
    compressed_like = generate_biocode(iter(drifted), content_codes=True)[0]

    assert original["units"][0] != compressed_like["units"][0]
    assert (
        original["content_codes"]["CONTENT_MIXED_V0"]["iscc"]
        == compressed_like["content_codes"]["CONTENT_MIXED_V0"]["iscc"]
    )


def test_selected_mixed_content_code_deterministic():
    """The helper is deterministic for the same IMAGEWALK plane sequence."""
    planes = [make_plane(z=z, value=z * 17) for z in range(7)]

    assert _selected_mixed_content_code(planes) == _selected_mixed_content_code(planes)


def test_simprints_structure():
    """Simprints have correct structure when enabled."""
    planes = [make_plane(z=z) for z in range(3)]
    results = generate_biocode(iter(planes), simprints=True)
    assert "simprints" in results[0]
    assert "DATA_NONE_V0" in results[0]["simprints"]

    entries = results[0]["simprints"]["DATA_NONE_V0"]
    assert len(entries) == 3

    for i, entry in enumerate(entries):
        assert "simprint" in entry
        assert "offset" in entry
        assert "size" in entry
        assert entry["offset"] == i
        assert entry["size"] == 64 * 64 * 1  # uint8


def test_simprints_offset_sequential():
    """Plane offsets are sequential within a scene, reset per scene."""
    planes = [
        make_plane(scene_idx=0, z=0, c=0),
        make_plane(scene_idx=0, z=0, c=1),
        make_plane(scene_idx=0, z=1, c=0),
        make_plane(scene_idx=1, z=0, c=0),
        make_plane(scene_idx=1, z=0, c=1),
    ]
    results = generate_biocode(iter(planes), simprints=True)

    scene0_offsets = [e["offset"] for e in results[0]["simprints"]["DATA_NONE_V0"]]
    scene1_offsets = [e["offset"] for e in results[1]["simprints"]["DATA_NONE_V0"]]

    assert scene0_offsets == [0, 1, 2]
    assert scene1_offsets == [0, 1]


def test_simprints_size_matches_dtype():
    """Simprint size reflects plane byte size (Y * X * dtype_bytes)."""
    planes_u16 = [make_plane(shape=(128, 128), dtype=np.uint16)]
    results = generate_biocode(iter(planes_u16), simprints=True)
    entry = results[0]["simprints"]["DATA_NONE_V0"][0]
    assert entry["size"] == 128 * 128 * 2

    planes_f32 = [make_plane(shape=(32, 32), dtype=np.float32)]
    results = generate_biocode(iter(planes_f32), simprints=True)
    entry = results[0]["simprints"]["DATA_NONE_V0"][0]
    assert entry["size"] == 32 * 32 * 4


def test_deterministic_output():
    """Same input planes produce identical output."""

    def make_planes():
        return [make_plane(z=z, value=z * 10) for z in range(5)]

    r1 = generate_biocode(iter(make_planes()), simprints=True)
    r2 = generate_biocode(iter(make_planes()), simprints=True)
    assert r1 == r2


def test_different_data_different_codes():
    """Different pixel data produces different ISCC-SUM codes."""
    planes_a = [make_plane(value=0)]
    planes_b = [make_plane(value=255)]
    r_a = generate_biocode(iter(planes_a))
    r_b = generate_biocode(iter(planes_b))
    assert r_a[0]["iscc_code"] != r_b[0]["iscc_code"]


def test_empty_input():
    """Empty plane iterator returns empty results."""
    results = generate_biocode(iter([]))
    assert results == []


def test_simprint_base64_valid():
    """Simprint values are valid base64 strings."""
    import base64

    planes = [make_plane(z=z, shape=(256, 256)) for z in range(2)]
    results = generate_biocode(iter(planes), simprints=True)

    for entry in results[0]["simprints"]["DATA_NONE_V0"]:
        decoded = base64.b64decode(entry["simprint"])
        assert len(decoded) == 32  # 256-bit data code body
