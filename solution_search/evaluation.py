from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConstraintConfig, EvalConfig, ObjectiveConfig


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    stdout: str
    stderr: str
    returncode: int
    runtime_seconds: float
    combined_log_path: Path


def _convert_metric(raw: str) -> Any:
    raw = raw.strip()
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def run_evaluation(
    evaluation: EvalConfig,
    workspace_root: Path,
    run_dir: Path,
    env_overrides: dict[str, str],
    candidate_id: str,
) -> EvaluationResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    cwd = workspace_root / evaluation.cwd
    env = os.environ.copy()
    formatted_defaults = {
        key: str(value).format(workspace=str(workspace_root), candidate_id=candidate_id)
        for key, value in evaluation.default_env.items()
    }
    formatted_overrides = {
        key: str(value).format(workspace=str(workspace_root), candidate_id=candidate_id)
        for key, value in env_overrides.items()
    }
    env.update(formatted_defaults)
    env.update(formatted_overrides)
    env["CANDIDATE_ID"] = candidate_id
    env["WORKSPACE_ROOT"] = str(workspace_root)
    command = evaluation.command.format(workspace=str(workspace_root), candidate_id=candidate_id)
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        shell=True,
        capture_output=True,
        text=True,
        timeout=evaluation.timeout_seconds,
    )
    runtime_seconds = time.perf_counter() - started
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    combined_path = run_dir / "combined.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    combined = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
    combined_path.write_text(combined, encoding="utf-8")

    metrics: dict[str, Any] = {
        "returncode": proc.returncode,
        "runtime_seconds": runtime_seconds,
    }
    sources = {"combined": combined}
    for rel_path in evaluation.capture_files:
        path = cwd / rel_path
        if path.exists() and path.is_file():
            sources[rel_path] = path.read_text(encoding="utf-8", errors="replace")
    for pattern in evaluation.metric_patterns:
        haystack = sources.get(pattern.source, "")
        matches = list(re.finditer(pattern.regex, haystack, flags=re.MULTILINE))
        if not matches:
            continue
        match = matches[-1]
        for key, value in match.groupdict().items():
            if value is not None:
                metrics[key] = _convert_metric(value)
    return EvaluationResult(
        metrics=metrics,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        runtime_seconds=runtime_seconds,
        combined_log_path=combined_path,
    )


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "<=":
        return actual <= expected
    if op == "<":
        return actual < expected
    if op == ">=":
        return actual >= expected
    if op == ">":
        return actual > expected
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    raise ValueError(f"unsupported constraint op: {op}")


def score_metrics(metrics: dict[str, Any], objective: ObjectiveConfig) -> tuple[float, bool, list[str]]:
    violations: list[str] = []
    feasible = True
    if int(metrics.get("returncode", 1)) != 0:
        feasible = False
    primary_value = metrics.get(objective.primary_metric)
    if primary_value is None:
        return objective.missing_metric_penalty, False, [f"missing primary metric: {objective.primary_metric}"]
    primary_numeric = float(primary_value)
    score = primary_numeric if objective.direction == "min" else -primary_numeric
    if int(metrics.get("returncode", 1)) != 0:
        score += objective.failed_returncode_penalty
        violations.append(f"returncode={metrics.get('returncode')}")
    for constraint in objective.constraints:
        actual = metrics.get(constraint.metric)
        if actual is None or not _compare(actual, constraint.op, constraint.value):
            feasible = False if constraint.hard else feasible
            score += constraint.penalty
            violations.append(f"{constraint.metric} {constraint.op} {constraint.value} violated (actual={actual})")
    return score, feasible, violations
