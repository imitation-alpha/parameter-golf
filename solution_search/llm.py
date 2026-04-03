from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        while start != -1:
            depth = 0
            for idx in range(start, len(text)):
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = text[start : idx + 1]
                        try:
                            return json.loads(snippet)
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
    raise LLMError("could not parse JSON object from model response")


class BaseLLM:
    def generate_json(self, schema_name: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate_json(self, schema_name: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.config.api_base:
            raise LLMError("api_base is required for openai_compatible provider")
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise LLMError(f"missing API key in env var {self.config.api_key_env}")
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = self.config.api_base.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            message = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM HTTP error {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise LLMError(f"LLM request failed: {exc}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected LLM response: {result}") from exc
        return _extract_json_object(content)


@dataclass
class StubState:
    proposal_calls: int = 0
    reflection_calls: int = 0
    knowledge_calls: int = 0


class StubLLM(BaseLLM):
    def __init__(self) -> None:
        self.state = StubState()

    def generate_json(self, schema_name: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if schema_name == "proposal_batch":
            self.state.proposal_calls += 1
            idx = self.state.proposal_calls
            return {
                "candidates": [
                    {
                        "title": f"stub_noop_candidate_{idx}",
                        "hypothesis": "Smoke-test the search loop without mutating code.",
                        "reasoning": "Use a no-op candidate to verify proposal, evaluation, archive, reflection, and branching plumbing.",
                        "strategy_id": "smoke-noop",
                        "parent_candidate_id": "baseline",
                        "env_overrides": {},
                        "edits": [],
                        "expected_metrics": {},
                        "tags": ["stub", "smoke"],
                    }
                ]
            }
        if schema_name == "reflection":
            self.state.reflection_calls += 1
            return {
                "summary": "Stub reflection: the run completed and the loop is functioning.",
                "outcome": "neutral",
                "lessons": ["The orchestration path executed end to end."],
                "next_hunches": ["Replace the stub provider with a real model and enable non-trivial edits."],
                "strategy_updates": [
                    {
                        "action": "reinforce",
                        "strategy_id": "smoke-noop",
                        "title": "Smoke no-op",
                        "description": "Use a no-op candidate to test orchestration before real search.",
                        "when_to_try": "Before first live API run or after infrastructure changes.",
                        "avoid_when": "When you need meaningful model improvements.",
                    }
                ],
                "mistake_updates": [],
            }
        if schema_name == "knowledge_summary":
            self.state.knowledge_calls += 1
            return {
                "title": f"stub_knowledge_{self.state.knowledge_calls}",
                "summary": "Stub knowledge summary for infrastructure testing.",
                "key_points": ["External knowledge ingestion path works."],
                "tactical_ideas": ["Use real model-backed summaries for live campaigns."],
                "cautions": ["Stub summaries are not semantically meaningful."],
                "tags": ["stub", "knowledge"],
            }
        raise LLMError(f"unknown schema requested from stub: {schema_name}")


def create_llm(config: LLMConfig) -> BaseLLM:
    if config.provider == "stub":
        return StubLLM()
    if config.provider == "openai_compatible":
        return OpenAICompatibleLLM(config)
    raise LLMError(f"unsupported provider: {config.provider}")
