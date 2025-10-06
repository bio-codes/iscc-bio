"""Common data structures and utilities for bioimage plane processing."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Plane:
    """2D pixel plane with dimensional position metadata.

    Represents a single 2D slice extracted from multi-dimensional bioimage data
    (5D: Scene/Z/Channel/Time/YX), retaining its position in all dimensions.

    :ivar xy_array: 2D pixel data array with Y (rows) and X (columns) dimensions
    :ivar scene_idx: Scene or series index (0-based)
    :ivar z_depth: Z-stack depth position (0-based)
    :ivar c_channel: Channel index (0-based)
    :ivar t_time: Time point index (0-based)
    """

    xy_array: np.ndarray
    scene_idx: int
    z_depth: int
    c_channel: int
    t_time: int


def plane_to_canonical_bytes(plane):
    # type: (np.ndarray) -> bytes
    """Convert 2D plane to canonical big-endian byte representation.

    Converts pixel data to a deterministic byte sequence using big-endian byte order
    and C-order (row-major) flattening. Optimized for performance with NumPy's native
    byte conversion instead of struct.pack.

    :param plane: 2D NumPy array (Y, X dimensions)
    :return: Flattened pixel data in big-endian byte order
    :raises ValueError: If plane is not 2-dimensional
    """
    if plane.ndim != 2:
        raise ValueError(f"Expected 2D plane, got {plane.ndim}D")

    # Flatten plane in C-order (row-major: Y then X)
    flat = plane.flatten(order="C")

    # Use numpy's tobytes() with explicit big-endian conversion
    # This is MUCH faster than struct.pack for large arrays
    if flat.dtype.byteorder == ">" or (
        flat.dtype.byteorder == "=" and np.little_endian
    ):
        # Already big-endian or need to swap
        canonical_bytes = flat.astype(f">{flat.dtype.char}", copy=False).tobytes()
    else:
        # Convert to big-endian
        canonical_bytes = flat.astype(f">{flat.dtype.char}").tobytes()

    return canonical_bytes
