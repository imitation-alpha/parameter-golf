from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileEdit:
    kind: str
    path: str
    find: str | None = None
    replace: str | None = None
    anchor: str | None = None
    text: str | None = None
    content: str | None = None
    pattern: str | None = None
    count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileEdit":
        return cls(
            kind=str(data["kind"]),
            path=str(data["path"]),
            find=data.get("find"),
            replace=data.get("replace"),
            anchor=data.get("anchor"),
            text=data.get("text"),
            content=data.get("content"),
            pattern=data.get("pattern"),
            count=int(data["count"]) if data.get("count") is not None else None,
        )


@dataclass
class Proposal:
    title: str
    hypothesis: str
    reasoning: str
    strategy_id: str | None = None
    parent_candidate_id: str | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)
    edits: list[FileEdit] = field(default_factory=list)
    expected_metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Proposal":
        edits = [FileEdit.from_dict(item) for item in data.get("edits", [])]
        env_overrides = {str(k): str(v) for k, v in data.get("env_overrides", {}).items()}
        tags = [str(item) for item in data.get("tags", [])]
        expected_metrics = data.get("expected_metrics", {})
        return cls(
            title=str(data["title"]),
            hypothesis=str(data["hypothesis"]),
            reasoning=str(data["reasoning"]),
            strategy_id=str(data["strategy_id"]) if data.get("strategy_id") else None,
            parent_candidate_id=str(data["parent_candidate_id"]) if data.get("parent_candidate_id") else None,
            env_overrides=env_overrides,
            edits=edits,
            expected_metrics=expected_metrics,
            tags=tags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "hypothesis": self.hypothesis,
            "reasoning": self.reasoning,
            "strategy_id": self.strategy_id,
            "parent_candidate_id": self.parent_candidate_id,
            "env_overrides": self.env_overrides,
            "edits": [edit.__dict__ for edit in self.edits],
            "expected_metrics": self.expected_metrics,
            "tags": self.tags,
        }


@dataclass
class ProposalBatch:
    candidates: list[Proposal] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProposalBatch":
        proposals = [Proposal.from_dict(item) for item in data.get("candidates", [])]
        return cls(candidates=proposals)

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [candidate.to_dict() for candidate in self.candidates]}


@dataclass
class Reflection:
    summary: str
    outcome: str
    lessons: list[str] = field(default_factory=list)
    next_hunches: list[str] = field(default_factory=list)
    strategy_updates: list[dict[str, Any]] = field(default_factory=list)
    mistake_updates: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reflection":
        return cls(
            summary=str(data.get("summary", "")),
            outcome=str(data.get("outcome", "neutral")),
            lessons=[str(item) for item in data.get("lessons", [])],
            next_hunches=[str(item) for item in data.get("next_hunches", [])],
            strategy_updates=list(data.get("strategy_updates", [])),
            mistake_updates=list(data.get("mistake_updates", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "outcome": self.outcome,
            "lessons": self.lessons,
            "next_hunches": self.next_hunches,
            "strategy_updates": self.strategy_updates,
            "mistake_updates": self.mistake_updates,
        }
