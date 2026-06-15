"""Biocode generation — scene-level ISCC-SUM with optional per-plane simprints.

Generates one ISCC-SUM per scene from any IMAGEWALK plane iterator. Output format
matches the IsccEntry schema of the ISCC search API (https://search.iscc.id).
"""

import base64

import iscc_lib
import numpy as np
from PIL import Image

from iscc_bio.imagewalk.common import plane_to_canonical_bytes


def generate_biocode(
    planes,
    *,
    simprints=False,
    content_codes=False,
    bits=256,
):
    """Generate biocode (ISCC-SUM) from a plane iterator.

    Works with any IMAGEWALK backend (bioio, ngff, blitz). Produces one IsccEntry
    per scene, optionally with per-plane data-code simprints for granular
    similarity search and an additional Content-Code-Mixed sidecar for
    content-layer comparison that is stable for selected controlled drift cases.

    :param planes: Iterable of Plane objects in IMAGEWALK Z->C->T order
    :param simprints: Generate per-plane data-code simprints (DATA_NONE_V0)
    :param content_codes: Generate a sidecar Content-Code-Mixed unit from up to
        three deterministic IMAGEWALK-selected planes
    :param bits: Bit length for ISCC codes (default: 256)
    :return: List of dicts matching the IsccEntry schema
    """
    results = []
    current_scene = None
    data_hasher = None
    inst_hasher = None
    scene_simprints = []
    scene_planes = []
    plane_idx = 0

    for plane in planes:
        if plane.scene_idx != current_scene:
            if data_hasher is not None:
                entry = _finalize_scene(
                    data_hasher,
                    inst_hasher,
                    scene_simprints,
                    scene_planes,
                    content_codes,
                    bits,
                )
                results.append(entry)

            current_scene = plane.scene_idx
            data_hasher = iscc_lib.DataHasher()
            inst_hasher = iscc_lib.InstanceHasher()
            scene_simprints = []
            scene_planes = []
            plane_idx = 0

        canonical_bytes = plane_to_canonical_bytes(plane.xy_array)
        data_hasher.update(canonical_bytes)
        inst_hasher.update(canonical_bytes)
        if content_codes:
            scene_planes.append(plane)

        if simprints:
            plane_simprint = _plane_simprint(canonical_bytes, plane_idx)
            scene_simprints.append(plane_simprint)

        plane_idx += 1

    if data_hasher is not None:
        entry = _finalize_scene(
            data_hasher,
            inst_hasher,
            scene_simprints,
            scene_planes,
            content_codes,
            bits,
        )
        results.append(entry)

    return results


def _plane_simprint(canonical_bytes, plane_idx):
    """Generate a simprint entry for a single plane.

    :param canonical_bytes: Canonical byte representation of the plane
    :param plane_idx: Linear plane index in Z->C->T traversal order
    :return: Dict with simprint, offset, and size fields
    """
    plane_hasher = iscc_lib.DataHasher(canonical_bytes)
    plane_result = plane_hasher.finalize(bits=256)
    _, _, _, _, body_bytes = iscc_lib.iscc_decode(plane_result["iscc"])
    simprint_b64 = base64.b64encode(body_bytes).decode("ascii")
    return {
        "simprint": simprint_b64,
        "offset": plane_idx,
        "size": len(canonical_bytes),
    }


def _select_plane_indices(plane_count, max_planes=3):
    """Select up to ``max_planes`` deterministic offsets from IMAGEWALK order."""
    if plane_count <= 0:
        return []
    if plane_count <= max_planes:
        return list(range(plane_count))
    if max_planes == 1:
        return [0]
    if max_planes == 2:
        return [0, plane_count - 1]
    middle = (plane_count - 1) // 2
    return [0, middle, plane_count - 1]


def _plane_to_image_pixels(plane_array):
    """Normalize a 2D plane to the 32x32 grayscale input expected by ISCC Image-Code.

    Percentile clipping makes the sidecar stable under small codec-induced
    intensity drift while remaining deterministic.
    """
    if plane_array.ndim != 2:
        raise ValueError(f"Expected 2D plane, got {plane_array.ndim}D")

    image = np.asarray(plane_array, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        image_u8 = np.zeros(image.shape, dtype=np.uint8)
    else:
        normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
        image_u8 = np.rint(normalized * 255).astype(np.uint8)

    pil_image = Image.fromarray(image_u8, mode="L")
    resized = pil_image.resize((32, 32), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).reshape(-1).tolist()


def _selected_mixed_content_code(planes, *, max_planes=3, bits=256):
    """Generate a sidecar ISCC Content-Code-Mixed from IMAGEWALK-selected planes.

    The selected offsets are deterministic in IMAGEWALK traversal order: all planes
    for scenes with up to ``max_planes`` planes, otherwise first, middle, and last.
    ``gen_mixed_code_v0`` requires at least two Content-Codes, so single-plane
    scenes duplicate the one Image Content-Code internally. The sidecar records
    that derivation policy explicitly so consumers can reproduce it.
    """
    offsets = _select_plane_indices(len(planes), max_planes=max_planes)
    if not offsets:
        raise ValueError("cannot generate Content-Code-Mixed without planes")

    image_codes = []
    for offset in offsets:
        pixels = _plane_to_image_pixels(planes[offset].xy_array)
        image_codes.append(iscc_lib.gen_image_code_v0(pixels, bits=bits)["iscc"])

    mix_inputs = image_codes if len(image_codes) > 1 else image_codes * 2
    mixed_code = iscc_lib.gen_mixed_code_v0(mix_inputs, bits=bits)["iscc"]
    return {
        "iscc": mixed_code,
        "unit_type": "CONTENT_MIXED_V0",
        "derivation": "IMAGEWALK_MIXED_CONTENT_V0",
        "bits": bits,
        "plane_count": len(planes),
        "selected_count": len(offsets),
        "offsets": offsets,
        "image_codes": image_codes,
        "mixed_input_count": len(mix_inputs),
    }


def _finalize_scene(
    data_hasher, inst_hasher, simprints_list, scene_planes, content_codes, bits
):
    """Finalize hashers and build an IsccEntry dict for one scene."""
    data_code = data_hasher.finalize(bits=bits)["iscc"]
    inst_code = inst_hasher.finalize(bits=bits)["iscc"]
    iscc_sum = iscc_lib.gen_iscc_code_v0([data_code, inst_code], wide=True)["iscc"]

    entry = {
        "iscc_code": iscc_sum,
        "units": [data_code, inst_code],
    }

    if simprints_list:
        entry["simprints"] = {"DATA_NONE_V0": simprints_list}

    if content_codes:
        entry["content_codes"] = {
            "CONTENT_MIXED_V0": _selected_mixed_content_code(scene_planes, bits=bits)
        }

    return entry
