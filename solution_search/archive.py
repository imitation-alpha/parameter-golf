from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Reflection


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=False)


class CampaignStore:
    def __init__(self, campaign_dir: Path, seed_strategies_path: Path | None = None, seed_mistakes_path: Path | None = None):
        self.root = campaign_dir
        self.archive_dir = self.root / "archive"
        self.candidates_dir = self.root / "candidates"
        self.runs_path = self.archive_dir / "runs.jsonl"
        self.strategies_path = self.archive_dir / "strategies.json"
        self.mistakes_path = self.archive_dir / "mistakes.json"
        self.meta_path = self.archive_dir / "meta.json"
        self.seed_strategies_path = seed_strategies_path
        self.seed_mistakes_path = seed_mistakes_path

    def ensure_initialized(self) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            _write_json(self.meta_path, {"created_at": _now_iso(), "next_candidate_index": 1})
        if not self.strategies_path.exists():
            strategies = _read_json(self.seed_strategies_path, []) if self.seed_strategies_path else []
            _write_json(self.strategies_path, strategies)
        if not self.mistakes_path.exists():
            mistakes = _read_json(self.seed_mistakes_path, []) if self.seed_mistakes_path else []
            _write_json(self.mistakes_path, mistakes)
        if not self.runs_path.exists():
            self.runs_path.write_text("", encoding="utf-8")

    def next_candidate_id(self) -> str:
        meta = _read_json(self.meta_path, {"next_candidate_index": 1})
        idx = int(meta.get("next_candidate_index", 1))
        meta["next_candidate_index"] = idx + 1
        _write_json(self.meta_path, meta)
        return f"candidate_{idx:04d}"

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_dir / candidate_id

    def append_run(self, record: dict[str, Any]) -> None:
        self.runs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=False) + "\n")

    def load_runs(self) -> list[dict[str, Any]]:
        if not self.runs_path.exists():
            return []
        runs: list[dict[str, Any]] = []
        with self.runs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
        return runs

    def load_strategies(self) -> list[dict[str, Any]]:
        return _read_json(self.strategies_path, [])

    def save_strategies(self, strategies: list[dict[str, Any]]) -> None:
        _write_json(self.strategies_path, strategies)

    def load_mistakes(self) -> list[dict[str, Any]]:
        return _read_json(self.mistakes_path, [])

    def save_mistakes(self, mistakes: list[dict[str, Any]]) -> None:
        _write_json(self.mistakes_path, mistakes)

    def best_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        runs = [run for run in self.load_runs() if run.get("status") == "completed" and run.get("feasible")]
        return sorted(runs, key=lambda item: (float(item.get("score", 1e18)), item.get("created_at", "")))[:limit]

    def recent_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        runs = self.load_runs()
        return runs[-limit:]

    def recent_failures(self, limit: int = 5) -> list[dict[str, Any]]:
        failures = [run for run in self.load_runs() if run.get("status") != "completed" or not run.get("feasible")]
        return failures[-limit:]

    def recent_lessons(self, limit: int = 6) -> list[str]:
        out: list[str] = []
        for run in reversed(self.load_runs()):
            for lesson in reversed(run.get("reflection", {}).get("lessons", [])):
                out.append(str(lesson))
                if len(out) >= limit:
                    return list(reversed(out))
        return list(reversed(out))

    def sample_strategies(self, count: int, rng: random.Random) -> list[dict[str, Any]]:
        strategies = self.load_strategies()
        if not strategies:
            return []
        weighted: list[tuple[float, dict[str, Any]]] = []
        for strategy in strategies:
            successes = float(strategy.get("successes", 0))
            failures = float(strategy.get("failures", 0))
            trials = successes + failures
            score = (successes + 1.0) / (trials + 2.0) + 1.0 / (1.0 + trials)
            weighted.append((score, strategy))
        selected: list[dict[str, Any]] = []
        pool = weighted[:]
        while pool and len(selected) < count:
            total = sum(score for score, _ in pool)
            needle = rng.random() * total
            acc = 0.0
            for idx, (score, strategy) in enumerate(pool):
                acc += score
                if acc >= needle:
                    selected.append(strategy)
                    pool.pop(idx)
                    break
        return selected

    def update_archives(
        self,
        proposal: dict[str, Any],
        reflection: Reflection,
        was_improvement: bool,
        run_id: str,
    ) -> None:
        strategies = self.load_strategies()
        strategy_by_id = {item.get("id"): item for item in strategies if item.get("id")}
        used_strategy_id = proposal.get("strategy_id")
        if used_strategy_id and used_strategy_id in strategy_by_id:
            key = "successes" if was_improvement else "failures"
            strategy_by_id[used_strategy_id][key] = int(strategy_by_id[used_strategy_id].get(key, 0)) + 1
            strategy_by_id[used_strategy_id]["last_outcome"] = reflection.outcome
            strategy_by_id[used_strategy_id]["last_seen_run_id"] = run_id
        for update in reflection.strategy_updates:
            action = str(update.get("action", "reinforce"))
            strategy_id = str(update.get("strategy_id") or "").strip()
            title = str(update.get("title") or strategy_id or "unnamed-strategy").strip()
            if not strategy_id:
                strategy_id = title.lower().replace(" ", "-")
            target = strategy_by_id.get(strategy_id)
            if target is None:
                target = {
                    "id": strategy_id,
                    "title": title,
                    "description": str(update.get("description", "")),
                    "when_to_try": str(update.get("when_to_try", "")),
                    "avoid_when": str(update.get("avoid_when", "")),
                    "tags": list(update.get("tags", [])),
                    "successes": 0,
                    "failures": 0,
                    "examples": [],
                }
                strategies.append(target)
                strategy_by_id[strategy_id] = target
            if action == "reinforce":
                target["successes"] = int(target.get("successes", 0)) + 1
            elif action == "deprioritize":
                target["failures"] = int(target.get("failures", 0)) + 1
            target["description"] = str(update.get("description", target.get("description", "")))
            target["when_to_try"] = str(update.get("when_to_try", target.get("when_to_try", "")))
            target["avoid_when"] = str(update.get("avoid_when", target.get("avoid_when", "")))
            target["last_update_run_id"] = run_id
        self.save_strategies(strategies)

        mistakes = self.load_mistakes()
        mistake_by_id = {item.get("id"): item for item in mistakes if item.get("id")}
        for update in reflection.mistake_updates:
            title = str(update.get("title", "")).strip()
            if not title:
                continue
            mistake_id = str(update.get("id") or title.lower().replace(" ", "-"))
            target = mistake_by_id.get(mistake_id)
            if target is None:
                target = {
                    "id": mistake_id,
                    "title": title,
                    "description": str(update.get("description", "")),
                    "avoidance": str(update.get("avoidance", "")),
                    "count": 0,
                    "examples": [],
                }
                mistakes.append(target)
                mistake_by_id[mistake_id] = target
            target["count"] = int(target.get("count", 0)) + 1
            target["description"] = str(update.get("description", target.get("description", "")))
            target["avoidance"] = str(update.get("avoidance", target.get("avoidance", "")))
            target["last_seen_run_id"] = run_id
        self.save_mistakes(mistakes)

