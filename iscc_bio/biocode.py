"""Biocode generation — scene-level ISCC-SUM with optional per-plane simprints.

Generates one ISCC-SUM per scene from any IMAGEWALK plane iterator. Output format
matches the IsccEntry schema of the ISCC search API (https://search.iscc.id).
"""

import base64
import iscc_lib
from iscc_bio.imagewalk.common import plane_to_canonical_bytes


def generate_biocode(
    planes,
    *,
    simprints=False,
    bits=256,
):
    # type: (Iterable[Plane], ..., bool, int) -> List[Dict[str, Any]]
    """Generate biocode (ISCC-SUM) from a plane iterator.

    Works with any IMAGEWALK backend (bioio, ngff, blitz). Produces one IsccEntry
    per scene, optionally with per-plane data-code simprints for granular similarity search.

    :param planes: Iterable of Plane objects in IMAGEWALK Z->C->T order
    :param simprints: Generate per-plane data-code simprints (DATA_NONE_V0)
    :param bits: Bit length for ISCC codes (default: 256)
    :return: List of dicts matching the IsccEntry schema
    """
    results = []
    current_scene = None
    data_hasher = None
    inst_hasher = None
    scene_simprints = []  # type: List[Dict[str, Any]]
    plane_idx = 0

    for plane in planes:
        if plane.scene_idx != current_scene:
            if data_hasher is not None:
                entry = _finalize_scene(data_hasher, inst_hasher, scene_simprints, bits)
                results.append(entry)

            current_scene = plane.scene_idx
            data_hasher = iscc_lib.DataHasher()
            inst_hasher = iscc_lib.InstanceHasher()
            scene_simprints = []
            plane_idx = 0

        canonical_bytes = plane_to_canonical_bytes(plane.xy_array)
        data_hasher.update(canonical_bytes)
        inst_hasher.update(canonical_bytes)

        if simprints:
            plane_simprint = _plane_simprint(canonical_bytes, plane_idx)
            scene_simprints.append(plane_simprint)

        plane_idx += 1

    if data_hasher is not None:
        entry = _finalize_scene(data_hasher, inst_hasher, scene_simprints, bits)
        results.append(entry)

    return results


def _plane_simprint(canonical_bytes, plane_idx):
    # type: (bytes, int) -> Dict[str, Any]
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


def _finalize_scene(data_hasher, inst_hasher, simprints_list, bits):
    # type: (Any, Any, List[Dict[str, Any]], int) -> Dict[str, Any]
    """Finalize hashers and build an IsccEntry dict for one scene."""
    data_code = data_hasher.finalize(bits=bits)["iscc"]
    inst_code = inst_hasher.finalize(bits=bits)["iscc"]
    iscc_sum = iscc_lib.gen_iscc_code_v0([data_code, inst_code], wide=True)["iscc"]

    entry = {
        "iscc_code": iscc_sum,
        "units": [data_code, inst_code],
    }  # type: Dict[str, Any]

    if simprints_list:
        entry["simprints"] = {"DATA_NONE_V0": simprints_list}

    return entry
