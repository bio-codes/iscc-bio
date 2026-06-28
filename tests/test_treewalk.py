"""Conformance tests for the TREEWALK deterministic traversal port.

Each test maps to a rule or test case in the TREEWALK specification. Tests use
real temporary directories (not mocks). Cases that depend on unicode NFC/NFD
duplicates on disk are intentionally avoided because case/normalization-folding
filesystems (Windows, macOS) cannot represent them.
"""

import os

import pytest

from iscc_bio.treewalk import listdir, treewalk, treewalk_ignore, treewalk_iscc


def _rel(paths, root):
    """Return yielded paths as root-relative POSIX strings, in yield order."""
    root = root.resolve()
    return [p.resolve().relative_to(root).as_posix() for p in paths]


def test_listdir_byte_ordering(tmp_path):
    """Spec 4.1: entries sort by raw UTF-8 bytes, so uppercase precedes lowercase."""
    for name in ["apple", "Banana", "cherry"]:
        (tmp_path / name).write_text("x")
    names = [e.name for e in listdir(tmp_path)]
    # 'B' (0x42) < 'a' (0x61) < 'c' (0x63) — a byte sort, not a locale/case sort
    assert names == ["Banana", "apple", "cherry"]


def test_ignore_files_yielded_first(tmp_path):
    """Spec 4.2 / Test Case 3: ignore files come first, each group sorted."""
    for name in ["zzz.txt", "aaa.txt", ".gitignore", ".npmignore"]:
        (tmp_path / name).write_text("x")
    names = [p.name for p in treewalk(tmp_path)]
    assert names == [".gitignore", ".npmignore", "aaa.txt", "zzz.txt"]


def test_nested_depth_first_order(tmp_path):
    """Spec 4.2: files before subdirectories, subdirectories recursed in order."""
    (tmp_path / "a" / "nested").mkdir(parents=True)
    (tmp_path / "b").mkdir()
    (tmp_path / "root.txt").write_text("x")
    (tmp_path / "a" / "file_a.txt").write_text("x")
    (tmp_path / "b" / "file_b.txt").write_text("x")
    (tmp_path / "a" / "nested" / "deep.txt").write_text("x")
    assert _rel(treewalk(tmp_path), tmp_path) == [
        "root.txt",
        "a/file_a.txt",
        "a/nested/deep.txt",
        "b/file_b.txt",
    ]


def test_directories_not_yielded(tmp_path):
    """Spec 4.2: containers are traversed but never appear in the output."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.txt").write_text("x")
    assert _rel(treewalk(tmp_path), tmp_path) == ["sub/f.txt"]


def test_gitignore_pattern_filtering(tmp_path):
    """Test Case 4: '*.log' filters logs; the ignore file itself is kept."""
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "app.py").write_text("x")
    (tmp_path / "debug.log").write_text("x")
    (tmp_path / "error.log").write_text("x")
    assert _rel(treewalk_ignore(tmp_path, ".gitignore"), tmp_path) == [
        ".gitignore",
        "app.py",
    ]


def test_cascading_ignore_reinclude(tmp_path):
    """Spec 5: child patterns override parent patterns ('!important.log')."""
    (tmp_path / "sub").mkdir()
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "sub" / ".gitignore").write_text("!important.log\n")
    (tmp_path / "app.py").write_text("x")
    (tmp_path / "debug.log").write_text("x")
    (tmp_path / "sub" / "error.log").write_text("x")
    (tmp_path / "sub" / "important.log").write_text("x")
    (tmp_path / "sub" / "data.txt").write_text("x")
    names = {p.name for p in treewalk_ignore(tmp_path, ".gitignore")}
    assert "app.py" in names
    assert "data.txt" in names
    assert "important.log" in names  # re-included by child .gitignore
    assert "debug.log" not in names  # ignored by root pattern
    assert "error.log" not in names  # ignored by inherited pattern


def test_treewalk_iscc_filters_metadata(tmp_path):
    """Section 6 / Test Case 5: drop *.iscc.json sidecars and .isccignore matches."""
    (tmp_path / ".isccignore").write_text("temp/\n*.bak\n")
    (tmp_path / "data.txt").write_text("x")
    (tmp_path / "data.txt.iscc.json").write_text("x")  # ISCC sidecar -> dropped
    (tmp_path / "backup.bak").write_text("x")  # matches *.bak -> dropped
    (tmp_path / "temp").mkdir()
    (tmp_path / "temp" / "cache.dat").write_text("x")  # temp/ not recursed
    assert _rel(treewalk_iscc(tmp_path), tmp_path) == [".isccignore", "data.txt"]


def test_iscc_sidecar_exclusion_not_overridable(tmp_path):
    """Section 6.2: *.iscc.json is always excluded, even if .isccignore re-includes."""
    (tmp_path / ".isccignore").write_text("!data.txt.iscc.json\n")
    (tmp_path / "data.txt").write_text("x")
    (tmp_path / "data.txt.iscc.json").write_text("x")
    assert _rel(treewalk_iscc(tmp_path), tmp_path) == [".isccignore", "data.txt"]


def test_dir_only_pattern_prunes_subtree(tmp_path):
    """Spec 5: a 'build/' directory pattern prunes the whole subtree.

    Upstream iscc-sum matches a directory via ``rel`` OR ``rel + '/'`` and stops
    descending, so a child ignore file inside the pruned directory is never read.
    A per-file rewrite would instead recurse and re-include ``build/artifact.o``.
    """
    (tmp_path / ".isccignore").write_text("build/\n")
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / ".isccignore").write_text("!artifact.o\n")
    (tmp_path / "build" / "artifact.o").write_text("x")
    assert _rel(treewalk_iscc(tmp_path), tmp_path) == [".isccignore", "a.txt"]


def test_dir_only_pattern_excludes_matching_file(tmp_path):
    """Spec 5: 'secret/' also excludes a regular file named 'secret'.

    The upstream predicate ORs in the trailing-slash form, so a file whose name
    equals a directory-only pattern is still excluded — matching the standard
    tool's byte-level file set.
    """
    (tmp_path / ".isccignore").write_text("secret/\n")
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "secret").write_text("x")  # a file, not a directory
    assert _rel(treewalk_iscc(tmp_path), tmp_path) == [".isccignore", "a.txt"]


def test_wildcard_ignore_does_not_reinclude_via_dir_negation(tmp_path):
    """Spec 5: '*' ignores everything; '!keep/' does not re-include keep's files.

    A non-conformant per-directory rewrite would keep ``keep/data.txt``; the
    upstream predicate prunes ``keep`` and yields nothing, like the standard tool.
    """
    (tmp_path / ".isccignore").write_text("*\n!keep/\n")
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "data.txt").write_text("x")
    (tmp_path / "other.txt").write_text("x")
    assert _rel(treewalk_iscc(tmp_path), tmp_path) == []


def test_symlinks_excluded(tmp_path):
    """Spec 4.3: references (symlinks) must not be followed or yielded."""
    (tmp_path / "real.txt").write_text("x")
    try:
        os.symlink(tmp_path / "real.txt", tmp_path / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")
    assert _rel(treewalk(tmp_path), tmp_path) == ["real.txt"]
