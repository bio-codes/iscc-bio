# -*- coding: utf-8 -*-
"""BioIO implementation of IMAGEWALK plane traversal.

This module provides deterministic plane traversal for multi-dimensional bioimage data
using the BioIO library, conforming to the IMAGEWALK specification.
"""

from iscc_bio.imagewalk.models import Plane
import bioio


def iter_planes_bioio(image):
    # type: (bioio.ImageLike) -> object
    """Iterate over 2D planes in a bioimage following IMAGEWALK Z→C→T traversal order.

    Processes each scene independently and yields planes in deterministic order:
    - Outermost loop: Z dimension (depth/focal plane)
    - Middle loop: C dimension (channel)
    - Innermost loop: T dimension (time)

    Conforms to IMAGEWALK specification for deterministic bioimage traversal.

    :param image: Path to bioimage file, fsspec URI, or array-like object
    :return: Generator yielding Plane objects in Z→C→T order
    """

    # Open image with BioIO
    img = bioio.BioImage(image)

    # Process each scene
    num_scenes = len(img.scenes)

    for scene_idx in range(num_scenes):
        # Set current scene
        if num_scenes > 1:
            img.set_scene(scene_idx)

        # Get dimension information
        dims = img.dims
        shape = dims.shape
        dim_order = dims.order

        # Find dimension indices
        t_idx = dim_order.index("T") if "T" in dim_order else None
        c_idx = dim_order.index("C") if "C" in dim_order else None
        z_idx = dim_order.index("Z") if "Z" in dim_order else None

        # Get dimension sizes (Y and X are required, others optional)
        size_t = shape[t_idx] if t_idx is not None else 1
        size_c = shape[c_idx] if c_idx is not None else 1
        size_z = shape[z_idx] if z_idx is not None else 1

        # Traverse planes in Z→C→T order (IMAGEWALK specification)
        for z in range(size_z):
            for c in range(size_c):
                for t in range(size_t):
                    # Build kwargs for dimension selection
                    kwargs = {}
                    if z_idx is not None:
                        kwargs["Z"] = z
                    if c_idx is not None:
                        kwargs["C"] = c
                    if t_idx is not None:
                        kwargs["T"] = t

                    # Get 2D plane (YX dimensions)
                    xy_array = img.get_image_data("YX", **kwargs)

                    # Yield Plane object
                    yield Plane(
                        xy_array=xy_array,
                        scene_idx=scene_idx,
                        z_depth=z,
                        c_channel=c,
                        t_time=t,
                    )


if __name__ == "__main__":
    pass
