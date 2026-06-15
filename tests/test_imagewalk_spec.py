# -*- coding: utf-8 -*-
"""Test suite for IMAGEWALK specification compliance.

Tests the canonical byte conversion and traversal order requirements
as defined in the IMAGEWALK specification.
"""

import numpy as np
import pytest
from iscc_bio.imagewalk import plane_to_canonical_bytes, Plane


class TestCanonicalByteConversion:
    """Test canonical byte conversion per IMAGEWALK spec Section 7.1."""

    def test_case1_tiny_uint8_plane(self):
        # type: () -> None
        """Test Case 1: Tiny uint8 Plane."""
        # Input: Y=2, X=2, uint8, [[1, 2], [3, 4]]
        plane = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        # Expected: 0x01 0x02 0x03 0x04
        expected = b"\x01\x02\x03\x04"

        result = plane_to_canonical_bytes(plane)
        assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    def test_case2_tiny_uint16_plane(self):
        # type: () -> None
        """Test Case 2: Tiny uint16 Plane."""
        # Input: Y=2, X=2, uint16, [[256, 512], [768, 1024]]
        plane = np.array([[256, 512], [768, 1024]], dtype=np.uint16)

        # Expected: 0x01 0x00  0x02 0x00  0x03 0x00  0x04 0x00 (big-endian)
        expected = b"\x01\x00\x02\x00\x03\x00\x04\x00"

        result = plane_to_canonical_bytes(plane)
        assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    def test_case3_float32_plane(self):
        # type: () -> None
        """Test Case 3: Float32 Plane."""
        # Input: Y=1, X=2, float32, [[1.0, 2.0]]
        plane = np.array([[1.0, 2.0]], dtype=np.float32)

        # Expected: IEEE 754 single precision, big-endian
        # 1.0 = 0x3F800000, 2.0 = 0x40000000
        expected = b"\x3f\x80\x00\x00\x40\x00\x00\x00"

        result = plane_to_canonical_bytes(plane)
        assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    def test_row_major_order(self):
        # type: () -> None
        """Test that flattening follows row-major order."""
        # Create a 3x3 array with unique values
        plane = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)

        # Expected: row 0, then row 1, then row 2
        expected = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09"

        result = plane_to_canonical_bytes(plane)
        assert result == expected, "Flattening must follow row-major order"

    def test_int16_big_endian(self):
        # type: () -> None
        """Test int16 big-endian encoding."""
        # Test negative numbers too
        plane = np.array([[-1, 1000]], dtype=np.int16)

        # -1 in int16 big-endian = 0xFFFF
        # 1000 in int16 big-endian = 0x03E8
        expected = b"\xff\xff\x03\xe8"

        result = plane_to_canonical_bytes(plane)
        assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    def test_float64_plane(self):
        # type: () -> None
        """Test float64 (double precision) encoding."""
        plane = np.array([[1.0]], dtype=np.float64)

        # 1.0 in float64 big-endian = 0x3FF0000000000000
        expected = b"\x3f\xf0\x00\x00\x00\x00\x00\x00"

        result = plane_to_canonical_bytes(plane)
        assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    def test_uint32_plane(self):
        # type: () -> None
        """Test uint32 big-endian encoding."""
        plane = np.array([[0x12345678]], dtype=np.uint32)

        # 0x12345678 in big-endian = 0x12 0x34 0x56 0x78
        expected = b"\x12\x34\x56\x78"

        result = plane_to_canonical_bytes(plane)
        assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    def test_only_2d_accepted(self):
        # type: () -> None
        """Test that only 2D arrays are accepted (no auto-squeezing)."""
        # 2D array should work
        plane_2d = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = plane_to_canonical_bytes(plane_2d)
        assert result == b"\x01\x02\x03\x04"

        # Arrays with extra dimensions should be rejected
        plane_4d = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D plane"):
            plane_to_canonical_bytes(plane_4d)

    def test_non_2d_raises_error(self):
        # type: () -> None
        """Test that non-2D arrays raise an error."""
        # 1D array
        plane_1d = np.array([1, 2, 3], dtype=np.uint8)

        with pytest.raises(ValueError, match="Expected 2D plane"):
            plane_to_canonical_bytes(plane_1d)

        # 3D array
        plane_3d = np.ones((2, 2, 2), dtype=np.uint8)

        with pytest.raises(ValueError, match="Expected 2D plane"):
            plane_to_canonical_bytes(plane_3d)

    def test_lazy_array_like_without_flatten(self):
        # type: () -> None
        """Test that array-likes lacking .flatten() are materialized.

        Some bioio backends (e.g. the bffile-based bioio-bioformats >=2 reader)
        yield a lazy array-like from .compute() that exposes shape/dtype/ndim but
        has no .flatten(). Such input must be coerced to NumPy via np.asarray().
        """

        class FakeLazyArray:
            """Minimal LazyBioArray stand-in: no .flatten(), but __array__-able."""

            def __init__(self, arr):
                # type: (np.ndarray) -> None
                self._arr = arr
                self.ndim = arr.ndim
                self.dtype = arr.dtype
                self.shape = arr.shape

            def __array__(self, dtype=None):
                return self._arr if dtype is None else self._arr.astype(dtype)

        plane = np.array([[256, 512], [768, 1024]], dtype=np.uint16)
        lazy = FakeLazyArray(plane)

        assert not hasattr(lazy, "flatten")
        # Coerced result must match the materialized big-endian bytes.
        assert plane_to_canonical_bytes(lazy) == plane_to_canonical_bytes(plane)
        assert plane_to_canonical_bytes(lazy) == b"\x01\x00\x02\x00\x03\x00\x04\x00"

    def test_non_2d_lazy_array_like_raises_before_materializing(self):
        # type: () -> None
        """Test that the ndim guard rejects non-2D lazy input without a read.

        The guard runs before coercion and relies on the array-like exposing
        .ndim, so non-2D input is rejected without triggering __array__.
        """

        class ExplodingLazyArray:
            """Reports ndim but raises if anything tries to materialize it."""

            def __init__(self, ndim):
                # type: (int) -> None
                self.ndim = ndim

            def __array__(self, dtype=None):
                raise AssertionError("must not materialize before ndim guard")

        with pytest.raises(ValueError, match="Expected 2D plane"):
            plane_to_canonical_bytes(ExplodingLazyArray(3))


class TestTraversalOrder:
    """Test traversal order requirements per IMAGEWALK spec Section 4.3."""

    def test_traversal_order_matches_spec(self):
        # type: () -> None
        """Test Case 4: Multi-Dimensional Traversal Order."""
        # Create planes with T=2, C=2, Z=2
        size_t, size_c, size_z = 2, 2, 2

        # Expected order per spec
        expected_order = [
            (0, 0, 0),  # z=0, c=0, t=0
            (0, 0, 1),  # z=0, c=0, t=1
            (0, 1, 0),  # z=0, c=1, t=0
            (0, 1, 1),  # z=0, c=1, t=1
            (1, 0, 0),  # z=1, c=0, t=0
            (1, 0, 1),  # z=1, c=0, t=1
            (1, 1, 0),  # z=1, c=1, t=0
            (1, 1, 1),  # z=1, c=1, t=1
        ]

        # Generate actual order using nested loops
        actual_order = []
        for z in range(size_z):  # Outermost
            for c in range(size_c):  # Middle
                for t in range(size_t):  # Innermost
                    actual_order.append((z, c, t))

        assert actual_order == expected_order, "Traversal must follow Z→C→T order"

    def test_plane_dataclass(self):
        # type: () -> None
        """Test Plane dataclass structure."""
        xy_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        plane = Plane(xy_array=xy_array, scene_idx=0, z_depth=1, c_channel=2, t_time=3)

        assert plane.xy_array is xy_array
        assert plane.scene_idx == 0
        assert plane.z_depth == 1
        assert plane.c_channel == 2
        assert plane.t_time == 3


class TestHighLevelAPI:
    """Test high-level IMAGEWALK API functions."""

    def test_imagewalk_bioio_exists(self):
        # type: () -> None
        """Test that imagewalk_bioio function exists and is callable."""
        from iscc_bio.imagewalk import imagewalk_bioio

        assert callable(imagewalk_bioio)

    def test_imagewalk_ngff_exists(self):
        # type: () -> None
        """Test that imagewalk_ngff function exists and is callable."""
        from iscc_bio.imagewalk import imagewalk_ngff

        assert callable(imagewalk_ngff)

    def test_imagewalk_blitz_image_exists(self):
        # type: () -> None
        """Test that imagewalk_blitz_image function exists and is callable."""
        from iscc_bio.imagewalk import imagewalk_blitz_image

        assert callable(imagewalk_blitz_image)

    def test_imagewalk_blitz_fileset_exists(self):
        # type: () -> None
        """Test that imagewalk_blitz_fileset function exists and is callable."""
        from iscc_bio.imagewalk import imagewalk_blitz_fileset

        assert callable(imagewalk_blitz_fileset)


class TestDataTypeSupport:
    """Test support for all required data types per spec Section 6.2."""

    def test_all_required_types(self):
        # type: () -> None
        """Test that all required data types are supported."""
        required_types = [
            (np.uint8, b"\x01"),
            (np.uint16, b"\x00\x01"),
            (np.uint32, b"\x00\x00\x00\x01"),
            (np.int8, b"\x01"),
            (np.int16, b"\x00\x01"),
            (np.int32, b"\x00\x00\x00\x01"),
            (np.float32, b"\x3f\x80\x00\x00"),  # 1.0
            (np.float64, b"\x3f\xf0\x00\x00\x00\x00\x00\x00"),  # 1.0
        ]

        for dtype, expected_bytes in required_types:
            if dtype in (np.float32, np.float64):
                plane = np.array([[1.0]], dtype=dtype)
            else:
                plane = np.array([[1]], dtype=dtype)

            result = plane_to_canonical_bytes(plane)
            assert result == expected_bytes, f"Failed for dtype {dtype}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
