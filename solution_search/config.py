from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConstraintConfig:
    metric: str
    op: str
    value: float | int | str
    penalty: float = 1_000_000.0
    hard: bool = True


@dataclass
class ObjectiveConfig:
    primary_metric: str
    direction: str = "min"
    constraints: list[ConstraintConfig] = field(default_factory=list)
    improvement_epsilon: float = 0.0
    failed_returncode_penalty: float = 10_000_000.0
    missing_metric_penalty: float = 10_000_000.0


@dataclass
class WorkspaceConfig:
    source_root: Path
    editable_paths: list[str]
    extra_copy_paths: list[str] = field(default_factory=list)
    ignore_paths: list[str] = field(default_factory=lambda: [".git", "__pycache__", "solution_search_runs"])
    campaign_dir: Path = Path("solution_search_runs/default")
    keep_workspaces: bool = False
    preflight_py_compile: bool = True


@dataclass
class MetricPatternConfig:
    regex: str
    source: str = "combined"


@dataclass
class EvalConfig:
    command: str
    cwd: str = "."
    timeout_seconds: int = 900
    default_env: dict[str, str] = field(default_factory=dict)
    metric_patterns: list[MetricPatternConfig] = field(default_factory=list)
    capture_files: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    api_base: str | None = None
    api_key_env: str = "LLM_API_KEY"
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4_000
    timeout_seconds: int = 120


@dataclass
class KnowledgeSourceConfig:
    id: str
    kind: str
    url: str | None = None
    repo: str | None = None
    path: str | None = None
    query: str | None = None
    max_chars: int = 40_000
    max_results: int = 3
    refresh_seconds: int = 86_400


@dataclass
class SearchConfig:
    iterations: int = 5
    evaluate_baseline: bool = True
    top_context_candidates: int = 3
    recent_failures: int = 4
    strategy_samples: int = 4
    recent_lessons: int = 6
    max_file_context_chars: int = 80_000
    random_seed: int = 0
    branch_factor: int = 1
    parent_pool_size: int = 1
    exploration_weight: float = 1.2
    refresh_knowledge_on_start: bool = True


@dataclass
class CampaignConfig:
    name: str
    problem_statement: str
    workspace: WorkspaceConfig
    evaluation: EvalConfig
    objective: ObjectiveConfig
    llm: LLMConfig
    search: SearchConfig
    knowledge_sources: list[KnowledgeSourceConfig] = field(default_factory=list)
    seed_strategies_path: Path | None = None
    seed_mistakes_path: Path | None = None
    source_notes: list[str] = field(default_factory=list)


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_campaign_config(path: str | Path) -> CampaignConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    base_dir = config_path.parent

    workspace_data = data["workspace"]
    workspace = WorkspaceConfig(
        source_root=_resolve_path(base_dir, workspace_data["source_root"]) or base_dir,
        editable_paths=[str(item) for item in workspace_data["editable_paths"]],
        extra_copy_paths=[str(item) for item in workspace_data.get("extra_copy_paths", [])],
        ignore_paths=[str(item) for item in workspace_data.get("ignore_paths", [".git", "__pycache__", "solution_search_runs"])],
        campaign_dir=_resolve_path(base_dir, workspace_data.get("campaign_dir", "solution_search_runs/default")) or Path("solution_search_runs/default"),
        keep_workspaces=bool(workspace_data.get("keep_workspaces", False)),
        preflight_py_compile=bool(workspace_data.get("preflight_py_compile", True)),
    )

    metric_patterns = [
        MetricPatternConfig(regex=str(item["regex"]), source=str(item.get("source", "combined")))
        for item in data["evaluation"].get("metric_patterns", [])
    ]
    evaluation = EvalConfig(
        command=str(data["evaluation"]["command"]),
        cwd=str(data["evaluation"].get("cwd", ".")),
        timeout_seconds=int(data["evaluation"].get("timeout_seconds", 900)),
        default_env={str(k): str(v) for k, v in data["evaluation"].get("default_env", {}).items()},
        metric_patterns=metric_patterns,
        capture_files=[str(item) for item in data["evaluation"].get("capture_files", [])],
    )

    constraints = [
        ConstraintConfig(
            metric=str(item["metric"]),
            op=str(item["op"]),
            value=item["value"],
            penalty=float(item.get("penalty", 1_000_000.0)),
            hard=bool(item.get("hard", True)),
        )
        for item in data["objective"].get("constraints", [])
    ]
    objective = ObjectiveConfig(
        primary_metric=str(data["objective"]["primary_metric"]),
        direction=str(data["objective"].get("direction", "min")),
        constraints=constraints,
        improvement_epsilon=float(data["objective"].get("improvement_epsilon", 0.0)),
        failed_returncode_penalty=float(data["objective"].get("failed_returncode_penalty", 10_000_000.0)),
        missing_metric_penalty=float(data["objective"].get("missing_metric_penalty", 10_000_000.0)),
    )

    llm = LLMConfig(
        provider=str(data["llm"].get("provider", "openai_compatible")),
        api_base=str(data["llm"]["api_base"]) if data["llm"].get("api_base") else None,
        api_key_env=str(data["llm"].get("api_key_env", "LLM_API_KEY")),
        model=str(data["llm"].get("model", "")),
        temperature=float(data["llm"].get("temperature", 0.7)),
        max_tokens=int(data["llm"].get("max_tokens", 4_000)),
        timeout_seconds=int(data["llm"].get("timeout_seconds", 120)),
    )

    search = SearchConfig(
        iterations=int(data["search"].get("iterations", 5)),
        evaluate_baseline=bool(data["search"].get("evaluate_baseline", True)),
        top_context_candidates=int(data["search"].get("top_context_candidates", 3)),
        recent_failures=int(data["search"].get("recent_failures", 4)),
        strategy_samples=int(data["search"].get("strategy_samples", 4)),
        recent_lessons=int(data["search"].get("recent_lessons", 6)),
        max_file_context_chars=int(data["search"].get("max_file_context_chars", 80_000)),
        random_seed=int(data["search"].get("random_seed", 0)),
        branch_factor=int(data["search"].get("branch_factor", 1)),
        parent_pool_size=int(data["search"].get("parent_pool_size", 1)),
        exploration_weight=float(data["search"].get("exploration_weight", 1.2)),
        refresh_knowledge_on_start=bool(data["search"].get("refresh_knowledge_on_start", True)),
    )

    knowledge_sources = [
        KnowledgeSourceConfig(
            id=str(item["id"]),
            kind=str(item["kind"]),
            url=str(item["url"]) if item.get("url") else None,
            repo=str(item["repo"]) if item.get("repo") else None,
            path=str(item["path"]) if item.get("path") else None,
            query=str(item["query"]) if item.get("query") else None,
            max_chars=int(item.get("max_chars", 40_000)),
            max_results=int(item.get("max_results", 3)),
            refresh_seconds=int(item.get("refresh_seconds", 86_400)),
        )
        for item in data.get("knowledge_sources", [])
    ]

    return CampaignConfig(
        name=str(data["name"]),
        problem_statement=str(data["problem_statement"]),
        workspace=workspace,
        evaluation=evaluation,
        objective=objective,
        llm=llm,
        search=search,
        knowledge_sources=knowledge_sources,
        seed_strategies_path=_resolve_path(base_dir, data.get("seed_strategies_path")),
        seed_mistakes_path=_resolve_path(base_dir, data.get("seed_mistakes_path")),
        source_notes=[str(item) for item in data.get("source_notes", [])],
    )
