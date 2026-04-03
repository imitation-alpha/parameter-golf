from __future__ import annotations

import os
import py_compile
import shutil
from pathlib import Path


def _is_rel_equal_or_parent(parent: str, child: str) -> bool:
    if parent == "":
        return True
    parent_parts = Path(parent).parts
    child_parts = Path(child).parts
    return len(parent_parts) <= len(child_parts) and child_parts[: len(parent_parts)] == parent_parts


def _should_copy(rel_path: str, copied_paths: set[str]) -> bool:
    return rel_path in copied_paths


def _subtree_contains_copy(rel_path: str, copied_paths: set[str]) -> bool:
    return any(_is_rel_equal_or_parent(rel_path, candidate) for candidate in copied_paths)


def _is_ignored(rel_path: str, ignore_paths: set[str]) -> bool:
    return any(_is_rel_equal_or_parent(ignore_path, rel_path) for ignore_path in ignore_paths)


def materialize_workspace(source_root: Path, workspace_root: Path, copied_paths: list[str], ignore_paths: list[str]) -> None:
    copied = {Path(path).as_posix() for path in copied_paths}
    ignored = {Path(path).as_posix() for path in ignore_paths}
    workspace_root.mkdir(parents=True, exist_ok=True)

    def walk(src: Path, dst: Path, rel_path: str) -> None:
        if rel_path and _is_ignored(rel_path, ignored):
            return
        if rel_path and not _subtree_contains_copy(rel_path, copied):
            os.symlink(src, dst, target_is_directory=src.is_dir())
            return
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        dst.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir(), key=lambda item: item.name):
            child_rel = child.relative_to(source_root).as_posix()
            child_dst = dst / child.name
            walk(child, child_dst, child_rel)

    for child in sorted(source_root.iterdir(), key=lambda item: item.name):
        rel = child.relative_to(source_root).as_posix()
        walk(child, workspace_root / child.name, rel)


def snapshot_editable_paths(workspace_root: Path, snapshot_root: Path, editable_paths: list[str]) -> None:
    for rel_path in editable_paths:
        src = workspace_root / rel_path
        dst = snapshot_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def overlay_snapshot(snapshot_root: Path, workspace_root: Path) -> None:
    if not snapshot_root.exists():
        return
    for path in sorted(snapshot_root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(snapshot_root)
        dst = workspace_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def py_compile_paths(workspace_root: Path, editable_paths: list[str]) -> list[str]:
    failures: list[str] = []
    for rel_path in editable_paths:
        path = workspace_root / rel_path
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:  # pragma: no cover - exercised in real runs
            failures.append(f"{rel_path}: {exc.msg}")
    return failures

