from __future__ import annotations

import difflib
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .archive import CampaignStore
from .config import CampaignConfig, load_campaign_config
from .edits import EditApplicationError, apply_edits
from .evaluation import run_evaluation, score_metrics
from .knowledge import KnowledgeManager
from .llm import LLMError, create_llm
from .models import Proposal, ProposalBatch, Reflection
from .prompts import (
    PROPOSAL_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    build_proposal_user_prompt,
    build_reflection_user_prompt,
)
from .workspace import materialize_workspace, overlay_snapshot, py_compile_paths, snapshot_editable_paths


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class SearchEngine:
    def __init__(self, config_path: str | Path):
        self.config = load_campaign_config(config_path)
        self.store = CampaignStore(
            self.config.workspace.campaign_dir,
            seed_strategies_path=self.config.seed_strategies_path,
            seed_mistakes_path=self.config.seed_mistakes_path,
        )
        self.store.ensure_initialized()
        self.llm = create_llm(self.config.llm)
        self.knowledge = KnowledgeManager(self.config.workspace.campaign_dir, self.config, self.llm)
        self.rng = random.Random(self.config.search.random_seed)

    def run(self, iterations: int | None = None) -> list[dict[str, Any]]:
        requested_iterations = iterations if iterations is not None else self.config.search.iterations
        runs: list[dict[str, Any]] = []
        if self.config.search.refresh_knowledge_on_start:
            self.refresh_knowledge(force=False)
        if self.config.search.evaluate_baseline and not self._has_baseline():
            runs.append(self._evaluate_baseline())
        for index in range(requested_iterations):
            runs.extend(self._run_iteration(index + 1))
        return runs

    def leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.store.best_runs(limit=limit)

    def refresh_knowledge(self, force: bool = False) -> list[dict[str, Any]]:
        return self.knowledge.refresh(force=force)

    def _has_baseline(self) -> bool:
        return any(run.get("candidate_id") == "baseline" for run in self.store.load_runs())

    def _default_parent(self) -> dict[str, Any] | None:
        best = self.store.best_runs(limit=1)
        if best:
            return best[0]
        runs = self.store.load_runs()
        for run in runs:
            if run.get("candidate_id") == "baseline":
                return run
        return None

    def _source_snapshot_files(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rel_path in self.config.workspace.editable_paths:
            out[rel_path] = _read_text(self.config.workspace.source_root / rel_path)
        return out

    def _load_parent_files(self, parent: dict[str, Any] | None) -> dict[str, str]:
        if parent is None or not parent.get("snapshot_dir"):
            return self._source_snapshot_files()
        snapshot_root = Path(parent["snapshot_dir"])
        if not snapshot_root.exists():
            return self._source_snapshot_files()
        out: dict[str, str] = {}
        for rel_path in self.config.workspace.editable_paths:
            snap_path = snapshot_root / rel_path
            out[rel_path] = _read_text(snap_path)
        return out

    def _evaluate_baseline(self) -> dict[str, Any]:
        return self._execute_candidate(
            candidate_id="baseline",
            proposal=Proposal(
                title="baseline",
                hypothesis="Evaluate the unmodified starting point.",
                reasoning="Baseline evaluation anchors the search archive.",
            ),
            parent=None,
            iteration=0,
        )

    def _run_iteration(self, iteration: int) -> list[dict[str, Any]]:
        parents = self._select_parent_pool()
        proposals = self._propose_batch(iteration, parents)
        records: list[dict[str, Any]] = []
        run_lookup = {run["candidate_id"]: run for run in self.store.load_runs()}
        for proposal in proposals:
            parent = run_lookup.get(proposal.parent_candidate_id) if proposal.parent_candidate_id else None
            if parent is None:
                parent = parents[0] if parents else self._default_parent()
            candidate_id = self.store.next_candidate_id()
            records.append(self._execute_candidate(candidate_id=candidate_id, proposal=proposal, parent=parent, iteration=iteration))
            run_lookup[candidate_id] = records[-1]
        return records

    def _select_parent_pool(self) -> list[dict[str, Any]]:
        runs = self.store.load_runs()
        if not runs:
            return []
        by_id = {run["candidate_id"]: run for run in runs}
        child_counts: dict[str, int] = defaultdict(int)
        for run in runs:
            parent_id = run.get("parent_candidate_id")
            if parent_id:
                child_counts[str(parent_id)] += 1
        feasible = [run for run in runs if run.get("feasible")]
        if not feasible:
            feasible = [run for run in runs if run.get("candidate_id") == "baseline"] or runs[:1]
        total_children = sum(child_counts.values()) + len(feasible)
        scored: list[tuple[float, dict[str, Any]]] = []
        for run in feasible:
            visits = 1 + child_counts.get(run["candidate_id"], 0)
            exploitation = -float(run.get("score", 1e18))
            exploration = self.config.search.exploration_weight * math.sqrt(math.log(total_children + 1.0) / visits)
            scored.append((exploitation + exploration, run))
        scored.sort(key=lambda item: item[0], reverse=True)
        limit = max(1, self.config.search.parent_pool_size)
        return [run for _, run in scored[:limit]]

    def _propose_batch(self, iteration: int, parents: list[dict[str, Any]]) -> list[Proposal]:
        if not parents:
            default_parent = self._default_parent()
            parents = [default_parent] if default_parent else []
        parent_files = {
            (parent.get("candidate_id") if parent else "baseline"): self._load_parent_files(parent)
            for parent in parents
        }
        best_runs = self.store.best_runs(limit=self.config.search.top_context_candidates)
        recent_failures = self.store.recent_failures(limit=self.config.search.recent_failures)
        strategies = self.store.sample_strategies(self.config.search.strategy_samples, self.rng)
        mistakes = self.store.load_mistakes()[-self.config.search.recent_failures :]
        lessons = self.store.recent_lessons(limit=self.config.search.recent_lessons)
        external_knowledge = self.knowledge.format_cards(limit=8)
        prompt = build_proposal_user_prompt(
            self.config,
            iteration,
            parent_candidates=parents,
            parent_file_contexts=parent_files,
            best_runs=best_runs,
            recent_failures=recent_failures,
            strategies=strategies,
            mistakes=mistakes,
            recent_lessons=lessons,
            external_knowledge=external_knowledge,
            branch_factor=self.config.search.branch_factor,
        )
        try:
            batch_dict = self.llm.generate_json("proposal_batch", PROPOSAL_SYSTEM_PROMPT, prompt)
            batch = ProposalBatch.from_dict(batch_dict)
        except Exception:
            batch = ProposalBatch(candidates=[])
        allowed_parent_ids = {parent.get("candidate_id") for parent in parents if parent is not None}
        fallback_parent_id = parents[0].get("candidate_id") if parents else None
        proposals: list[Proposal] = []
        for proposal in batch.candidates[: self.config.search.branch_factor]:
            if proposal.parent_candidate_id not in allowed_parent_ids:
                proposal.parent_candidate_id = fallback_parent_id
            proposals.append(proposal)
        while len(proposals) < self.config.search.branch_factor:
            proposals.append(
                Proposal(
                    title=f"fallback_noop_{len(proposals)+1}",
                    hypothesis="Fallback candidate because the proposal batch was too small.",
                    reasoning="Keep the branching loop alive even when the model under-produces candidates.",
                    parent_candidate_id=fallback_parent_id,
                    tags=["fallback", "noop"],
                )
            )
        return proposals

    def _execute_candidate(
        self,
        candidate_id: str,
        proposal: Proposal,
        parent: dict[str, Any] | None,
        iteration: int,
    ) -> dict[str, Any]:
        candidate_dir = self.store.candidate_dir(candidate_id)
        workspace_root = candidate_dir / "workspace"
        snapshot_root = candidate_dir / "snapshot"
        copied_paths = sorted(set(self.config.workspace.editable_paths + self.config.workspace.extra_copy_paths))
        materialize_workspace(
            self.config.workspace.source_root,
            workspace_root,
            copied_paths=copied_paths,
            ignore_paths=self.config.workspace.ignore_paths,
        )
        if parent is not None:
            overlay_snapshot(Path(parent["snapshot_dir"]), workspace_root)
        diff_summary = ""
        status = "completed"
        reflection = Reflection(summary="", outcome="neutral")
        syntax_failures: list[str] = []
        stdout_tail = ""
        stderr_tail = ""
        metrics: dict[str, Any]
        try:
            apply_edits(workspace_root, self.config.workspace.editable_paths, proposal.edits)
            snapshot_editable_paths(workspace_root, snapshot_root, self.config.workspace.editable_paths)
            diff_summary = self._compute_diff(parent, snapshot_root)
            if self.config.workspace.preflight_py_compile:
                syntax_failures = py_compile_paths(workspace_root, self.config.workspace.editable_paths)
            if syntax_failures:
                metrics = {"returncode": 999, "runtime_seconds": 0.0, "syntax_failures": syntax_failures}
                combined_log_path = candidate_dir / "combined.log"
                combined_log_path.write_text("\n".join(syntax_failures), encoding="utf-8")
            else:
                result = run_evaluation(
                    self.config.evaluation,
                    workspace_root=workspace_root,
                    run_dir=candidate_dir,
                    env_overrides=proposal.env_overrides,
                    candidate_id=candidate_id,
                )
                metrics = result.metrics
                stdout_tail = "\n".join(result.stdout.splitlines()[-80:])
                stderr_tail = "\n".join(result.stderr.splitlines()[-80:])
                combined_log_path = result.combined_log_path
        except (EditApplicationError, LLMError, TimeoutError, RuntimeError) as exc:
            status = "failed"
            metrics = {"returncode": 998, "runtime_seconds": 0.0, "error": str(exc)}
            combined_log_path = candidate_dir / "combined.log"
            combined_log_path.write_text(str(exc), encoding="utf-8")

        score, feasible, violations = score_metrics(metrics, self.config.objective)
        previous_best = self._default_parent()
        previous_best_score = float(previous_best["score"]) if previous_best is not None else None
        was_improvement = feasible and previous_best_score is not None and score < (previous_best_score - self.config.objective.improvement_epsilon)
        if feasible and previous_best is None:
            was_improvement = True
        if not proposal.edits and not proposal.env_overrides:
            proposal.tags.append("noop")
        record = {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "created_at": _now_iso(),
            "status": status,
            "proposal": proposal.to_dict(),
            "parent_candidate_id": parent.get("candidate_id") if parent else None,
            "metrics": metrics,
            "primary_metric": self.config.objective.primary_metric,
            "score": score,
            "feasible": feasible,
            "violations": violations,
            "snapshot_dir": str(snapshot_root),
            "workspace_dir": str(workspace_root),
            "combined_log_path": str(combined_log_path),
        }
        log_fallback = ""
        if combined_log_path.exists():
            log_fallback = combined_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        reflection = self._reflect(proposal, record, diff_summary, previous_best, stdout_tail or stderr_tail or log_fallback)
        record["reflection"] = reflection.to_dict()
        self.store.append_run(record)
        self.store.update_archives(proposal.to_dict(), reflection, was_improvement, candidate_id)
        (candidate_dir / "proposal.json").write_text(json.dumps(proposal.to_dict(), indent=2), encoding="utf-8")
        (candidate_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if not self.config.workspace.keep_workspaces and workspace_root.exists():
            shutil.rmtree(workspace_root)
            record["workspace_dir"] = ""
            (candidate_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def _compute_diff(self, parent: dict[str, Any] | None, snapshot_root: Path) -> str:
        lines: list[str] = []
        for rel_path in self.config.workspace.editable_paths:
            after = (snapshot_root / rel_path).read_text(encoding="utf-8")
            if parent is not None and parent.get("snapshot_dir") and Path(parent["snapshot_dir"]).exists():
                before_path = Path(parent["snapshot_dir"]) / rel_path
                before = before_path.read_text(encoding="utf-8")
            else:
                before = (self.config.workspace.source_root / rel_path).read_text(encoding="utf-8")
            diff = list(
                difflib.unified_diff(
                    before.splitlines(),
                    after.splitlines(),
                    fromfile=f"before/{rel_path}",
                    tofile=f"after/{rel_path}",
                    lineterm="",
                )
            )
            if diff:
                lines.extend(diff[:400])
        return "\n".join(lines) if lines else "No diff."

    def _reflect(
        self,
        proposal: Proposal,
        record: dict[str, Any],
        diff_summary: str,
        previous_best: dict[str, Any] | None,
        log_tail: str,
    ) -> Reflection:
        prompt = build_reflection_user_prompt(
            self.config,
            proposal=proposal,
            result=record,
            diff_summary=diff_summary,
            previous_best=previous_best,
            log_tail=log_tail[-4000:],
        )
        try:
            reflection_dict = self.llm.generate_json("reflection", REFLECTION_SYSTEM_PROMPT, prompt)
            return Reflection.from_dict(reflection_dict)
        except Exception as exc:  # pragma: no cover - fallback path
            return Reflection(
                summary=f"Reflection fallback: {exc}",
                outcome="neutral",
                lessons=["Reflection model failed; inspect candidate logs manually."],
            )
