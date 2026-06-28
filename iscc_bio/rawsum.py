"""Raw-byte ISCC-SUM over a source file or directory tree.

Computes a decode-independent ISCC-SUM (Data- + Instance-Code) directly over the
original source bytes via ``iscc_lib.SumHasher``. A single file is streamed as-is;
a directory (e.g. an OME-Zarr store or a multi-file fileset) is traversed with
TREEWALK-ISCC for deterministic, cross-platform ordering, so the code is
reproducible regardless of host filesystem.

This is the top-level identity for a bioimage. Because it never decodes pixels it
is produced even when image reading fails, and a single-file result is identical
to the standard ``iscc-sum`` tool. Per-scene IMAGEWALK content codes are nested
under it as ``parts``.
"""

from pathlib import Path

import iscc_lib

from iscc_bio.treewalk import treewalk_iscc

IO_READ_SIZE = 2 * 1024 * 1024  # 2 MiB streaming chunks


def raw_sum(source, *, bits=256, wide=True):
    # type: (str|Path, int, bool) -> dict
    """Generate a raw-byte ISCC-SUM over a file or directory tree.

    For a directory, files are hashed in TREEWALK-ISCC order (ISCC sidecar
    metadata excluded), producing one tree-level code. For a single file the
    result is identical to ``iscc_lib.gen_sum_code_v0``.

    :param source: Path to a file or directory (e.g. an OME-Zarr store)
    :param bits: Bit length for the component units (default: 256)
    :param wide: Produce a wide (128-bit-per-unit) ISCC-SUM (default: True)
    :return: Dict with ``iscc_code``, ``units``, ``datahash``, ``filesize``
    """
    source = Path(source)
    hasher = iscc_lib.SumHasher()
    if source.is_dir():
        for file_path in treewalk_iscc(source):
            _stream_file(hasher, file_path)
    else:
        _stream_file(hasher, source)

    result = hasher.finalize(bits=bits, wide=wide, add_units=True)
    return {
        "iscc_code": result["iscc"],
        "units": result["units"],
        "datahash": result["datahash"],
        "filesize": result["filesize"],
    }


def _stream_file(hasher, file_path):
    # type: (iscc_lib.SumHasher, Path) -> None
    """Feed one file's bytes into the hasher in fixed-size chunks."""
    with open(file_path, "rb") as f:
        while chunk := f.read(IO_READ_SIZE):
            hasher.update(chunk)
