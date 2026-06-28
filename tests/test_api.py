"""Tests for the high-level biocode Python API."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from iscc_bio.api import biocode
from iscc_bio.biocode import generate_biocode
from iscc_bio.imagewalk.common import Plane


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


def test_no_args_raises_value_error():
    """Calling biocode() with no arguments raises ValueError."""
    with pytest.raises(ValueError, match="Provide"):
        biocode()


def test_nonexistent_file_raises_file_not_found():
    """Non-existent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        biocode("/nonexistent/path/file.tif")


def test_invalid_source_type_raises(tmp_path):
    """Invalid source_type raises ValueError."""
    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="Unknown source_type"):
        biocode(str(dummy), source_type="invalid")


def test_omero_without_iid_or_fid_raises():
    """OMERO mode without iid or fid raises ValueError."""
    with pytest.raises(ValueError, match="iid.*fid"):
        biocode(host="server.com", username="u", password="p")


def test_omero_without_credentials_raises():
    """OMERO mode without username/password raises ValueError."""
    with pytest.raises(ValueError, match="username.*password"):
        biocode(host="server.com", iid=1)


def test_omero_without_host_or_conn_raises():
    """OMERO with iid but no host or conn raises ValueError."""
    with pytest.raises(ValueError, match="host.*conn"):
        biocode(iid=1)


def test_ambiguous_source_and_omero_raises():
    """Specifying both source and OMERO params raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        biocode("file.tif", host="server.com", iid=1, username="u", password="p")


def test_source_and_fid_raises():
    """Specifying both source and fid raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        biocode("file.tif", fid=1)


def test_source_and_conn_raises():
    """Specifying both source and conn raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        biocode("file.tif", conn=MagicMock())


def test_local_file_delegates_to_bioio(tmp_path):
    """Local file path delegates to iter_planes_bioio and returns correct results."""
    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"dummy")
    planes = [make_plane(z=z, value=z * 10) for z in range(3)]
    expected = generate_biocode(iter(planes))

    fresh_planes = [make_plane(z=z, value=z * 10) for z in range(3)]
    with patch("iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(fresh_planes)):
        result = biocode(str(dummy))

    assert len(result["parts"]) == len(expected)
    assert result["parts"][0]["iscc_code"] == expected[0]["iscc_code"]
    assert result["parts"][0]["units"] == expected[0]["units"]
    assert result["iscc_code"].startswith("ISCC:")  # raw-byte top over the file
    assert result["generator"].startswith("iscc-bio - v")


def test_local_file_with_simprints(tmp_path):
    """Simprints flag is passed through to generate_biocode."""
    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"dummy")
    planes = [make_plane(z=z, value=z * 10) for z in range(3)]

    with patch("iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(planes)):
        result = biocode(str(dummy), simprints=True)

    part = result["parts"][0]
    assert "simprints" in part
    assert "DATA_NONE_V0" in part["simprints"]
    assert len(part["simprints"]["DATA_NONE_V0"]) == 3


def test_zarr_auto_detection_by_suffix(tmp_path):
    """Files with .zarr suffix use the NGFF iterator."""
    zarr_dir = tmp_path / "test.zarr"
    zarr_dir.mkdir()
    planes = [make_plane(value=42)]

    with patch(
        "iscc_bio.imagewalk.iter_planes_ngff", return_value=iter(planes)
    ) as mock:
        result = biocode(str(zarr_dir))

    mock.assert_called_once_with(str(zarr_dir))
    assert len(result["parts"]) == 1


def test_zarr_auto_detection_by_zattrs(tmp_path):
    """Directories with .zattrs file use the NGFF iterator."""
    zarr_dir = tmp_path / "dataset"
    zarr_dir.mkdir()
    (zarr_dir / ".zattrs").write_text("{}")
    planes = [make_plane(value=42)]

    with patch(
        "iscc_bio.imagewalk.iter_planes_ngff", return_value=iter(planes)
    ) as mock:
        result = biocode(str(zarr_dir))

    mock.assert_called_once_with(str(zarr_dir))
    assert len(result["parts"]) == 1
    # top-level code is the tree SUM over the store (here: the .zattrs file)
    assert result["iscc_code"].startswith("ISCC:")


def test_explicit_source_type_bioio(tmp_path):
    """Explicit source_type='bioio' forces BioIO iterator."""
    zarr_dir = tmp_path / "test.zarr"
    zarr_dir.mkdir()
    planes = [make_plane(value=42)]

    with patch(
        "iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(planes)
    ) as mock:
        biocode(str(zarr_dir), source_type="bioio")

    mock.assert_called_once_with(str(zarr_dir))


def test_explicit_source_type_zarr(tmp_path):
    """Explicit source_type='zarr' forces NGFF iterator."""
    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"dummy")
    planes = [make_plane(value=42)]

    with patch(
        "iscc_bio.imagewalk.iter_planes_ngff", return_value=iter(planes)
    ) as mock:
        biocode(str(dummy), source_type="zarr")

    mock.assert_called_once_with(str(dummy))


def test_pathlike_source(tmp_path):
    """Accepts Path objects (os.PathLike)."""
    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"dummy")
    planes = [make_plane(value=42)]

    with patch("iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(planes)):
        result = biocode(dummy)

    assert len(result["parts"]) == 1


def test_multi_scene_results(tmp_path):
    """Multi-scene data produces one entry per scene."""
    dummy = tmp_path / "multi.tif"
    dummy.write_bytes(b"dummy")
    planes = [
        make_plane(scene_idx=0, z=0, value=10),
        make_plane(scene_idx=0, z=1, value=20),
        make_plane(scene_idx=1, z=0, value=30),
    ]

    with patch("iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(planes)):
        result = biocode(str(dummy))

    assert len(result["parts"]) == 2
    assert result["parts"][0]["iscc_code"] != result["parts"][1]["iscc_code"]


def test_result_schema(tmp_path):
    """Container exposes a raw-byte top plus IsccEntry-schema parts."""
    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"dummy")
    planes = [make_plane(value=42)]

    with patch("iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(planes)):
        result = biocode(str(dummy))

    assert isinstance(result, dict)
    assert result["iscc_code"].startswith("ISCC:")
    assert isinstance(result["units"], list) and len(result["units"]) == 2
    assert result["generator"].startswith("iscc-bio - v")
    assert isinstance(result["parts"], list)

    entry = result["parts"][0]
    assert entry["iscc_code"].startswith("ISCC:")
    assert isinstance(entry["units"], list)
    assert len(entry["units"]) == 2
    assert all(u.startswith("ISCC:") for u in entry["units"])


def test_top_level_is_raw_byte_sum_of_source(tmp_path):
    """Top-level iscc_code is the raw-byte SUM of the source, independent of pixels."""
    from iscc_bio.rawsum import raw_sum

    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"some original file bytes")
    planes = [make_plane(value=42)]

    with patch("iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(planes)):
        result = biocode(str(dummy))

    assert result["iscc_code"] == raw_sum(dummy)["iscc_code"]


def test_decode_failure_still_returns_top_level(tmp_path):
    """Pixel decode failure still yields the raw-byte top-level code (degraded)."""
    dummy = tmp_path / "broken.tif"
    dummy.write_bytes(b"not a real image")

    def broken_planes(*args, **kwargs):
        raise RuntimeError("cannot decode")
        yield  # pragma: no cover - makes this a generator function

    with patch("iscc_bio.imagewalk.iter_planes_bioio", side_effect=broken_planes):
        result = biocode(str(dummy))

    assert result["iscc_code"].startswith("ISCC:")  # raw-byte top survived
    assert result["parts"] == []
    assert result["generator"].startswith("iscc-bio - v")


def test_deterministic_results(tmp_path):
    """Same input produces identical results across calls."""
    dummy = tmp_path / "test.tif"
    dummy.write_bytes(b"dummy")

    def make_planes():
        return [make_plane(z=z, value=z * 10) for z in range(5)]

    with patch(
        "iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(make_planes())
    ):
        r1 = biocode(str(dummy), simprints=True)

    with patch(
        "iscc_bio.imagewalk.iter_planes_bioio", return_value=iter(make_planes())
    ):
        r2 = biocode(str(dummy), simprints=True)

    assert r1 == r2
