"""Deterministic, platform-agnostic directory tree traversal (TREEWALK).

Ported from the ISCC TREEWALK specification and reference implementation in
``bio-codes/iscc-sum`` (spec DRAFT 2025-06-19). It produces a reproducible,
cross-platform file ordering so that a raw-byte ISCC-SUM over a multi-file
bioimage (e.g. an OME-Zarr store or a multi-file fileset) is identical
regardless of host filesystem or locale.

Keep this module in sync with the upstream TREEWALK spec to preserve
cross-implementation conformance.

Layers, each building on the previous:

1. ``listdir`` — deterministic entry sorting by NFC-normalized UTF-8 bytes
2. ``treewalk`` — TREEWALK-BASE traversal with ignore-file prioritization
3. ``treewalk_ignore`` — gitignore-style pattern filtering with cascading rules
4. ``treewalk_iscc`` — ISCC layer that also drops ``*.iscc.json`` metadata files

``treewalk_iscc`` matches the traversal used by the ``iscc-sum`` CLI, so tree
codes built on top of it are byte-compatible with the standard tool.
"""

import os
from pathlib import Path
from unicodedata import normalize

import pathspec


def listdir(path):
    # type: (str|Path) -> list[DirEntry]
    """List directory entries with deterministic cross-platform sorting.

    Entries are sorted by their NFC-normalized UTF-8 encoded names, with the
    original (pre-normalization) UTF-8 bytes as a tie-breaker for entries that
    normalize to the same name. Symlinks are excluded for security and
    consistency.

    :param path: Directory path to list
    :return: Sorted list of DirEntry objects (excluding symlinks)
    """
    with os.scandir(path) as it:
        filtered = [e for e in it if not e.is_symlink()]
    return sorted(
        filtered,
        key=lambda e: (
            normalize("NFC", e.name).encode("utf-8"),
            e.name.encode("utf-8"),
        ),
    )


def treewalk(path):
    # type: (str|Path) -> Iterator[Path]
    """Walk a directory tree and yield file paths in deterministic order.

    Yields, for each container depth-first:

    1. Ignore files (``.*ignore``) in the current directory, sorted
    2. Regular files in the current directory, sorted
    3. Files from subdirectories, recursively (subdirectories sorted)

    Directories themselves are never yielded. Symlinks are ignored.

    :param path: Directory path to walk
    :return: Iterator yielding Path objects for each file found
    """
    path = Path(path).resolve(strict=True)
    entries = listdir(path)
    dirs = [d for d in entries if d.is_dir()]
    files = [f for f in entries if f.is_file()]

    # First yield ignore files
    for file_entry in files:
        if file_entry.name.startswith(".") and file_entry.name.endswith("ignore"):
            yield Path(file_entry.path)

    # Then yield non-ignore files
    for file_entry in files:
        if not (file_entry.name.startswith(".") and file_entry.name.endswith("ignore")):
            yield Path(file_entry.path)

    # Then recurse into directories
    for dir_entry in dirs:
        yield from treewalk(Path(dir_entry.path))


def _is_ignored(fp, root_path, ignore_spec):
    # type: (Path, Path, pathspec.PathSpec|None) -> bool
    """Return True if a path is excluded by the accumulated ignore spec.

    Mirrors the upstream ``iscc-sum`` predicate: a path matches when either its
    POSIX-relative form or that form with a trailing slash matches, so a single
    rule covers files and directories identically and stays byte-conformant with
    the standard tool. ``as_posix`` keeps the trailing-slash form separator-stable
    across platforms.

    :param fp: File or directory path to test
    :param root_path: Root the patterns are relative to
    :param ignore_spec: Accumulated PathSpec, or None when no patterns apply
    :return: True if the path should be ignored
    """
    if ignore_spec is None:
        return False
    rel = fp.relative_to(root_path).as_posix()
    return ignore_spec.match_file(rel) or ignore_spec.match_file(rel + "/")


def treewalk_ignore(path, ignore_file_name, root_path=None, ignore_spec=None):
    # type: (str|Path, str, Path|None, pathspec.PathSpec|None) -> Iterator[Path]
    """Walk a directory tree while respecting gitignore-style patterns.

    Yields paths in the same deterministic order as ``treewalk`` while filtering
    on accumulated ignore patterns that cascade from the root down to each
    subdirectory. Child-directory patterns override parent patterns.

    :param path: Directory to walk
    :param ignore_file_name: Name of the ignore file to look for (e.g. '.gitignore')
    :param root_path: Root for relative path calculations (defaults to ``path``)
    :param ignore_spec: Existing PathSpec with patterns to extend
    :return: Iterator yielding Path objects for non-ignored files
    """
    path = Path(path).resolve(strict=True)
    if root_path is None:
        root_path = path
    else:
        root_path = Path(root_path).resolve(strict=True)

    # Load local ignore rules if present
    local_ignore = path / ignore_file_name
    if local_ignore.exists():
        with open(local_ignore, "r", encoding="utf-8") as f:
            new_spec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, f
            )
            ignore_spec = new_spec if ignore_spec is None else ignore_spec + new_spec

    entries = listdir(path)
    dirs = [d for d in entries if d.is_dir()]
    files = [f for f in entries if f.is_file()]

    # First yield ignore files (including the current one)
    for file_entry in files:
        file_path = Path(file_entry.path)
        if file_entry.name.startswith(".") and file_entry.name.endswith("ignore"):
            if not _is_ignored(file_path, root_path, ignore_spec):
                yield file_path

    # Then yield non-ignore files
    for file_entry in files:
        file_path = Path(file_entry.path)
        if not (file_entry.name.startswith(".") and file_entry.name.endswith("ignore")):
            if not _is_ignored(file_path, root_path, ignore_spec):
                yield file_path

    # Then recurse into directories
    for dir_entry in dirs:
        dir_path = Path(dir_entry.path)
        if not _is_ignored(dir_path, root_path, ignore_spec):
            yield from treewalk_ignore(
                dir_path, ignore_file_name, root_path, ignore_spec
            )


def treewalk_iscc(path):
    # type: (str|Path) -> Iterator[Path]
    """Walk a directory tree with ISCC-specific filtering.

    Builds on ``treewalk_ignore`` using ``.isccignore`` files and additionally
    drops ISCC sidecar metadata (files ending in ``.iscc.json``). The sidecar
    exclusion is unconditional and cannot be re-included via ``.isccignore``.

    :param path: Directory path to walk
    :return: Iterator yielding Path objects for non-ignored, non-metadata files
    """
    path = Path(path).resolve(strict=True)
    for file_path in treewalk_ignore(path, ".isccignore"):
        if not file_path.name.endswith(".iscc.json"):
            yield file_path
