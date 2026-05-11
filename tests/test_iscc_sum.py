"""Tests for scene-level ISCC-SUM generation."""

import numpy as np
from iscc_bio.imagewalk.common import Plane
from iscc_bio.iscc_sum import generate_iscc_sum


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
    results = generate_iscc_sum(iter(planes))
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
    results = generate_iscc_sum(iter(planes))
    assert len(results) == 3


def test_no_simprints_by_default():
    """Simprints are not included when flag is False."""
    planes = [make_plane(z=z) for z in range(3)]
    results = generate_iscc_sum(iter(planes))
    assert "simprints" not in results[0]


def test_simprints_structure():
    """Simprints have correct structure when enabled."""
    planes = [make_plane(z=z) for z in range(3)]
    results = generate_iscc_sum(iter(planes), simprints=True)
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
    results = generate_iscc_sum(iter(planes), simprints=True)

    scene0_offsets = [e["offset"] for e in results[0]["simprints"]["DATA_NONE_V0"]]
    scene1_offsets = [e["offset"] for e in results[1]["simprints"]["DATA_NONE_V0"]]

    assert scene0_offsets == [0, 1, 2]
    assert scene1_offsets == [0, 1]


def test_simprints_size_matches_dtype():
    """Simprint size reflects plane byte size (Y * X * dtype_bytes)."""
    planes_u16 = [make_plane(shape=(128, 128), dtype=np.uint16)]
    results = generate_iscc_sum(iter(planes_u16), simprints=True)
    entry = results[0]["simprints"]["DATA_NONE_V0"][0]
    assert entry["size"] == 128 * 128 * 2

    planes_f32 = [make_plane(shape=(32, 32), dtype=np.float32)]
    results = generate_iscc_sum(iter(planes_f32), simprints=True)
    entry = results[0]["simprints"]["DATA_NONE_V0"][0]
    assert entry["size"] == 32 * 32 * 4


def test_deterministic_output():
    """Same input planes produce identical output."""

    def make_planes():
        return [make_plane(z=z, value=z * 10) for z in range(5)]

    r1 = generate_iscc_sum(iter(make_planes()), simprints=True)
    r2 = generate_iscc_sum(iter(make_planes()), simprints=True)
    assert r1 == r2


def test_different_data_different_codes():
    """Different pixel data produces different ISCC-SUM codes."""
    planes_a = [make_plane(value=0)]
    planes_b = [make_plane(value=255)]
    r_a = generate_iscc_sum(iter(planes_a))
    r_b = generate_iscc_sum(iter(planes_b))
    assert r_a[0]["iscc_code"] != r_b[0]["iscc_code"]


def test_empty_input():
    """Empty plane iterator returns empty results."""
    results = generate_iscc_sum(iter([]))
    assert results == []


def test_simprint_base64_valid():
    """Simprint values are valid base64 strings."""
    import base64

    planes = [make_plane(z=z, shape=(256, 256)) for z in range(2)]
    results = generate_iscc_sum(iter(planes), simprints=True)

    for entry in results[0]["simprints"]["DATA_NONE_V0"]:
        decoded = base64.b64decode(entry["simprint"])
        assert len(decoded) == 32  # 256-bit data code body
