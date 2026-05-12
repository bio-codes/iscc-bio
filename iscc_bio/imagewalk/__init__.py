# -*- coding: utf-8 -*-
"""IMAGEWALK - Deterministic traversal of multi-dimensional bioimage pixel data.

This module implements the IMAGEWALK specification for producing consistent,
reproducible hash digests from bioimage data across platforms and file formats.
"""

from iscc_bio.imagewalk.common import Plane, plane_to_canonical_bytes
from iscc_bio.imagewalk.iw_bioio import iter_planes_bioio
from iscc_bio.imagewalk.iw_blitz import (
    iter_planes_blitz_image,
    iter_planes_blitz_fileset,
)
from iscc_bio.imagewalk.iw_ngff import iter_planes_ngff

__all__ = [
    "Plane",
    "plane_to_canonical_bytes",
    "iter_planes_bioio",
    "iter_planes_blitz_image",
    "iter_planes_blitz_fileset",
    "iter_planes_ngff",
    "imagewalk_bioio",
    "imagewalk_ngff",
    "imagewalk_blitz_image",
    "imagewalk_blitz_fileset",
]


def imagewalk_bioio(image, hash_class=None):
    # type: (object, type | None) -> list[str]
    """Generate IMAGEWALK hashes for a bioimage using BioIO.

    Implements the complete IMAGEWALK algorithm to produce one hash per scene/series.

    :param image: Path to bioimage file, fsspec URI, or array-like object
    :param hash_class: Hash processor class (must have update() and hexdigest() methods).
                       Defaults to hashlib.sha256 if not provided.
    :return: List of hash strings (one per scene/series)
    """
    if hash_class is None:
        import hashlib

        hash_class = hashlib.sha256

    hashes = []
    current_scene = None
    hasher = None

    for plane in iter_planes_bioio(image):
        # Initialize new hasher for each scene
        if plane.scene_idx != current_scene:
            if hasher is not None:
                # Finalize previous scene hash
                hashes.append(hasher.hexdigest())
            current_scene = plane.scene_idx
            hasher = hash_class()

        # Convert plane to canonical bytes and update hash
        canonical_bytes = plane_to_canonical_bytes(plane.xy_array)
        hasher.update(canonical_bytes)

    # Finalize last scene hash
    if hasher is not None:
        hashes.append(hasher.hexdigest())

    return hashes


def imagewalk_ngff(zarr_path, hash_class=None):
    # type: (str | object, type | None) -> list[str]
    """Generate IMAGEWALK hashes for an OME-NGFF/Zarr image.

    Implements the complete IMAGEWALK algorithm to produce one hash per scene/series.

    :param zarr_path: Path to OME-NGFF/Zarr file or directory
    :param hash_class: Hash processor class (must have update() and hexdigest() methods).
                       Defaults to hashlib.sha256 if not provided.
    :return: List of hash strings (one per scene/series)
    """
    if hash_class is None:
        import hashlib

        hash_class = hashlib.sha256

    hashes = []
    current_scene = None
    hasher = None

    for plane in iter_planes_ngff(zarr_path):
        # Initialize new hasher for each scene
        if plane.scene_idx != current_scene:
            if hasher is not None:
                # Finalize previous scene hash
                hashes.append(hasher.hexdigest())
            current_scene = plane.scene_idx
            hasher = hash_class()

        # Convert plane to canonical bytes and update hash
        canonical_bytes = plane_to_canonical_bytes(plane.xy_array)
        hasher.update(canonical_bytes)

    # Finalize last scene hash
    if hasher is not None:
        hashes.append(hasher.hexdigest())

    return hashes


def imagewalk_blitz_image(conn, image, hash_class=None):
    # type: (object, object, type | None) -> str
    """Generate IMAGEWALK hash for a single OMERO image.

    Implements the complete IMAGEWALK algorithm to produce a hash for one image/scene.

    :param conn: BlitzGateway connection to OMERO server
    :param image: OMERO Image object to process
    :param hash_class: Hash processor class (must have update() and hexdigest() methods).
                       Defaults to hashlib.sha256 if not provided.
    :return: Hash string for the image
    """
    if hash_class is None:
        import hashlib

        hash_class = hashlib.sha256

    hasher = hash_class()

    for plane in iter_planes_blitz_image(conn, image):
        # Convert plane to canonical bytes and update hash
        canonical_bytes = plane_to_canonical_bytes(plane.xy_array)
        hasher.update(canonical_bytes)

    return hasher.hexdigest()


def imagewalk_blitz_fileset(conn, fileset, hash_class=None):
    # type: (object, object, type | None) -> list[str]
    """Generate IMAGEWALK hashes for all images in an OMERO fileset.

    Implements the complete IMAGEWALK algorithm to produce one hash per image/scene.

    :param conn: BlitzGateway connection to OMERO server
    :param fileset: OMERO Fileset object to process
    :param hash_class: Hash processor class (must have update() and hexdigest() methods).
                       Defaults to hashlib.sha256 if not provided.
    :return: List of hash strings (one per image/scene in the fileset)
    """
    if hash_class is None:
        import hashlib

        hash_class = hashlib.sha256

    hashes = []
    current_scene = None
    hasher = None

    for plane in iter_planes_blitz_fileset(conn, fileset):
        # Initialize new hasher for each scene
        if plane.scene_idx != current_scene:
            if hasher is not None:
                # Finalize previous scene hash
                hashes.append(hasher.hexdigest())
            current_scene = plane.scene_idx
            hasher = hash_class()

        # Convert plane to canonical bytes and update hash
        canonical_bytes = plane_to_canonical_bytes(plane.xy_array)
        hasher.update(canonical_bytes)

    # Finalize last scene hash
    if hasher is not None:
        hashes.append(hasher.hexdigest())

    return hashes
