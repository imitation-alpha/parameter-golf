from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CampaignConfig
from .models import Proposal


PROPOSAL_SYSTEM_PROMPT = """You are an autonomous ML/code search worker.
Your job is to propose a small batch of candidate next experiments for improving the target system.

Rules:
- Return JSON only. No markdown.
- Prefer several coherent, diverse candidates over many noisy ones.
- You may mutate code and/or propose environment overrides.
- Only edit whitelisted files.
- Use the mistake archive to avoid repeating bad ideas.
- Use the external knowledge cards as hints, not as unquestioned truth.
- If changing only hyperparameters, you may return an empty edits list.
"""


REFLECTION_SYSTEM_PROMPT = """You are the reflection phase of an autonomous research loop.
Given a proposal, execution result, and logs, summarize what happened and update the search memory.

Rules:
- Return JSON only. No markdown.
- Distinguish clearly between wins, losses, and neutral outcomes.
- Add mistake updates only when the evidence is specific enough to prevent a repeated failure.
- Strategy updates should either reinforce useful approaches or deprioritize weak ones.
"""


def _format_runs(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "None."
    lines: list[str] = []
    for run in runs:
        metrics = run.get("metrics", {})
        lines.append(
            "- {id}: score={score:.6f} feasible={feasible} primary={primary} summary={summary}".format(
                id=run.get("candidate_id"),
                score=float(run.get("score", 0.0)),
                feasible=run.get("feasible"),
                primary=metrics.get(run.get("primary_metric", ""), metrics),
                summary=run.get("reflection", {}).get("summary", run.get("proposal", {}).get("title", "")),
            )
        )
    return "\n".join(lines)


def _format_strategies(strategies: list[dict[str, Any]]) -> str:
    if not strategies:
        return "None."
    lines = []
    for item in strategies:
        lines.append(
            "- {id}: {title}. when_to_try={when}. avoid_when={avoid}. stats={s}/{f}".format(
                id=item.get("id", ""),
                title=item.get("title", ""),
                when=item.get("when_to_try", ""),
                avoid=item.get("avoid_when", ""),
                s=item.get("successes", 0),
                f=item.get("failures", 0),
            )
        )
    return "\n".join(lines)


def _format_mistakes(mistakes: list[dict[str, Any]]) -> str:
    if not mistakes:
        return "None."
    lines = []
    for item in mistakes:
        lines.append(
            "- {title}: {description}. avoidance={avoidance}. seen={count}".format(
                title=item.get("title", ""),
                description=item.get("description", ""),
                avoidance=item.get("avoidance", ""),
                count=item.get("count", 0),
            )
        )
    return "\n".join(lines)


def build_proposal_user_prompt(
    config: CampaignConfig,
    iteration: int,
    parent_candidates: list[dict[str, Any]],
    parent_file_contexts: dict[str, dict[str, str]],
    best_runs: list[dict[str, Any]],
    recent_failures: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
    mistakes: list[dict[str, Any]],
    recent_lessons: list[str],
    external_knowledge: str,
    branch_factor: int,
) -> str:
    parent_sections: list[str] = []
    for parent in parent_candidates:
        parent_id = parent.get("candidate_id", "baseline")
        lines = [f"## PARENT: {parent_id}"]
        lines.append(
            "score={score:.6f} feasible={feasible} primary={primary}".format(
                score=float(parent.get("score", 0.0)),
                feasible=parent.get("feasible"),
                primary=parent.get("metrics", {}).get(parent.get("primary_metric", ""), parent.get("metrics", {})),
            )
        )
        file_contexts = parent_file_contexts.get(parent_id, {})
        for path, content in file_contexts.items():
            trimmed = content[: config.search.max_file_context_chars]
            lines.append(f"### FILE: {path}\n{trimmed}")
        parent_sections.append("\n".join(lines))
    editable_files_text = "\n\n".join(parent_sections)
    lessons = "\n".join(f"- {item}" for item in recent_lessons) if recent_lessons else "None."
    payload = {
        "candidates": [
            {
                "title": "short descriptive title",
                "hypothesis": "why this candidate might help",
                "reasoning": "concise decision process",
                "strategy_id": "existing strategy id or new short slug",
                "parent_candidate_id": "must be one of the provided parent ids",
                "env_overrides": {"OPTIONAL_ENV": "value"},
                "edits": [
                    {
                        "kind": "replace|insert_after|insert_before|append|write_file|regex_replace|insert_after_regex|insert_before_regex",
                        "path": config.workspace.editable_paths[0] if config.workspace.editable_paths else "path/to/file.py"
                    }
                ],
                "expected_metrics": {"metric_name": "expected direction"},
                "tags": ["short", "labels"]
            }
        ]
    }
    return f"""Problem:
{config.problem_statement}

Iteration:
{iteration}

Whitelisted editable files:
{json.dumps(config.workspace.editable_paths, indent=2)}

Default evaluation command:
{config.evaluation.command}

Default evaluation env:
{json.dumps(config.evaluation.default_env, indent=2, sort_keys=True)}

You must return exactly {branch_factor} candidates.

Allowed parent candidates:
{json.dumps([parent.get("candidate_id") for parent in parent_candidates], indent=2)}

Best runs so far:
{_format_runs(best_runs)}

Recent failed or infeasible runs:
{_format_runs(recent_failures)}

Sampled strategy cards:
{_format_strategies(strategies)}

Mistake archive:
{_format_mistakes(mistakes)}

Recent lessons:
{lessons}

External knowledge cards:
{external_knowledge}

Return shape:
{json.dumps(payload, indent=2)}

Parent candidate file contents:
{editable_files_text}
"""


def build_reflection_user_prompt(
    config: CampaignConfig,
    proposal: Proposal,
    result: dict[str, Any],
    diff_summary: str,
    previous_best: dict[str, Any] | None,
    log_tail: str,
) -> str:
    return f"""Problem:
{config.problem_statement}

Proposal:
{json.dumps(proposal.to_dict(), indent=2)}

Execution result:
{json.dumps(result, indent=2)}

Previous best feasible run:
{json.dumps(previous_best, indent=2) if previous_best else "None"}

Editable diff summary:
{diff_summary}

Log tail:
{log_tail}

Return JSON with:
{{
  "summary": "what happened",
  "outcome": "win|loss|neutral",
  "lessons": ["short lesson", "..."],
  "next_hunches": ["next idea", "..."],
  "strategy_updates": [
    {{
      "action": "reinforce|deprioritize|new",
      "strategy_id": "optional-existing-id",
      "title": "strategy title",
      "description": "what the strategy is",
      "when_to_try": "when to use it",
      "avoid_when": "when not to use it"
    }}
  ],
  "mistake_updates": [
    {{
      "id": "optional-slug",
      "title": "mistake title",
      "description": "what went wrong",
      "avoidance": "how to avoid repeating it"
    }}
  ]
}}
"""
