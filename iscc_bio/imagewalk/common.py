from dataclasses import dataclass
import numpy as np


@dataclass
class Plane:
    """Represents a 2D pixel plane with its position metadata.

    Attributes:
        xy_array: 2D NumPy array of pixel data (Y, X dimensions)
        scene_idx: Scene/series index
        z_depth: Z-stack position (0-based)
        c_channel: Channel index (0-based)
        t_time: Time point index (0-based)
    """

    xy_array: np.ndarray
    scene_idx: int
    z_depth: int
    c_channel: int
    t_time: int
