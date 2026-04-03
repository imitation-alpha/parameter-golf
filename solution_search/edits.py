from __future__ import annotations

import re
from pathlib import Path

from .models import FileEdit


class EditApplicationError(RuntimeError):
    pass


def _is_editable_path(rel_path: str, editable_paths: list[str]) -> bool:
    target = Path(rel_path).as_posix()
    for allowed in editable_paths:
        allowed_rel = Path(allowed).as_posix()
        if target == allowed_rel:
            return True
        allowed_parts = Path(allowed_rel).parts
        target_parts = Path(target).parts
        if len(allowed_parts) <= len(target_parts) and target_parts[: len(allowed_parts)] == allowed_parts:
            return True
    return False


def _replace_once(text: str, find: str, replace: str, count: int | None) -> str:
    occurrences = text.count(find)
    if occurrences == 0:
        raise EditApplicationError("replace target not found")
    desired = count if count is not None else 1
    if desired >= 0 and occurrences < desired:
        raise EditApplicationError(f"replace target found {occurrences} times, expected at least {desired}")
    if count is None and occurrences != 1:
        raise EditApplicationError(f"replace target found {occurrences} times, expected exactly 1")
    return text.replace(find, replace, desired if desired >= 0 else occurrences)


def _insert_around(text: str, anchor: str, payload: str, after: bool, count: int | None) -> str:
    occurrences = text.count(anchor)
    if occurrences == 0:
        raise EditApplicationError("insert anchor not found")
    if count is None and occurrences != 1:
        raise EditApplicationError(f"insert anchor found {occurrences} times, expected exactly 1")
    desired = count if count is not None else 1
    needle = anchor + payload if after else payload + anchor
    return text.replace(anchor, needle, desired)


def _regex_replace(text: str, pattern: str, replace: str, count: int | None) -> str:
    compiled = re.compile(pattern, flags=re.MULTILINE | re.DOTALL)
    occurrences = len(compiled.findall(text))
    if occurrences == 0:
        raise EditApplicationError("regex target not found")
    if count is None and occurrences != 1:
        raise EditApplicationError(f"regex target found {occurrences} times, expected exactly 1")
    desired = count if count is not None else 1
    return compiled.sub(replace, text, count=desired)


def _regex_insert(text: str, pattern: str, payload: str, after: bool, count: int | None) -> str:
    compiled = re.compile(pattern, flags=re.MULTILINE | re.DOTALL)
    matches = list(compiled.finditer(text))
    if not matches:
        raise EditApplicationError("regex insert anchor not found")
    if count is None and len(matches) != 1:
        raise EditApplicationError(f"regex insert anchor found {len(matches)} times, expected exactly 1")
    desired = count if count is not None else 1
    offset = 0
    for match in matches[:desired]:
        insert_at = match.end() if after else match.start()
        piece = payload
        text = text[: insert_at + offset] + piece + text[insert_at + offset :]
        offset += len(piece)
    return text


def apply_edits(workspace_root: Path, editable_paths: list[str], edits: list[FileEdit]) -> None:
    for edit in edits:
        if not _is_editable_path(edit.path, editable_paths):
            raise EditApplicationError(f"edit path is not editable: {edit.path}")
        path = workspace_root / edit.path
        if edit.kind == "write_file":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit.content or "", encoding="utf-8")
            continue
        text = path.read_text(encoding="utf-8")
        if edit.kind == "replace":
            if edit.find is None or edit.replace is None:
                raise EditApplicationError("replace edit requires find and replace")
            text = _replace_once(text, edit.find, edit.replace, edit.count)
        elif edit.kind == "insert_after":
            if edit.anchor is None or edit.text is None:
                raise EditApplicationError("insert_after edit requires anchor and text")
            text = _insert_around(text, edit.anchor, edit.text, after=True, count=edit.count)
        elif edit.kind == "insert_before":
            if edit.anchor is None or edit.text is None:
                raise EditApplicationError("insert_before edit requires anchor and text")
            text = _insert_around(text, edit.anchor, edit.text, after=False, count=edit.count)
        elif edit.kind == "append":
            if edit.text is None:
                raise EditApplicationError("append edit requires text")
            text = text + edit.text
        elif edit.kind == "regex_replace":
            if edit.pattern is None or edit.replace is None:
                raise EditApplicationError("regex_replace edit requires pattern and replace")
            text = _regex_replace(text, edit.pattern, edit.replace, edit.count)
        elif edit.kind == "insert_after_regex":
            if edit.pattern is None or edit.text is None:
                raise EditApplicationError("insert_after_regex edit requires pattern and text")
            text = _regex_insert(text, edit.pattern, edit.text, after=True, count=edit.count)
        elif edit.kind == "insert_before_regex":
            if edit.pattern is None or edit.text is None:
                raise EditApplicationError("insert_before_regex edit requires pattern and text")
            text = _regex_insert(text, edit.pattern, edit.text, after=False, count=edit.count)
        else:
            raise EditApplicationError(f"unsupported edit kind: {edit.kind}")
        path.write_text(text, encoding="utf-8")

