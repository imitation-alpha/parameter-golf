from __future__ import annotations

import html
import json
import posixpath
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .config import CampaignConfig, KnowledgeSourceConfig
from .llm import BaseLLM, LLMError


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
        json.dump(value, handle, indent=2)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        text = html.unescape("".join(self.parts))
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def _strip_markup_noise(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`{3,}.*?`{3,}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _heuristic_summary(source: KnowledgeSourceConfig, content: str) -> dict[str, Any]:
    cleaned = _strip_markup_noise(content)
    lines = [line.strip() for line in cleaned.splitlines()]
    title = source.id
    for line in lines:
        if not line:
            continue
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            break
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            break
        title = line[:120]
        break
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    summary_parts: list[str] = []
    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            continue
        summary_parts.append(paragraph)
        if len(" ".join(summary_parts)) >= 420:
            break
    summary = " ".join(summary_parts)[:700] if summary_parts else cleaned[:700]
    bullets = [
        line.lstrip("-* ").strip()
        for line in lines
        if line.startswith(("- ", "* ")) and len(line.lstrip("-* ").strip()) > 8
    ]
    cautions = [
        bullet for bullet in bullets
        if any(token in bullet.lower() for token in ("risk", "warning", "avoid", "collapse", "unstable", "constraint"))
    ][:3]
    return {
        "title": title,
        "summary": summary,
        "key_points": bullets[:4],
        "tactical_ideas": [],
        "cautions": cautions,
        "tags": [source.kind, "heuristic-fallback"],
    }


def _fetch_url(url: str, timeout: int = 30) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "parameter-golf-solution-search/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read().decode("utf-8", errors="replace")
    return content_type, body


def _github_readme_candidates(repo: str) -> list[str]:
    owner, name = repo.split("/", 1)
    urls: list[str] = []
    for branch in ("main", "master"):
        for readme in ("README.md", "readme.md", "README.MD"):
            urls.append(f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{readme}")
    return urls


def _fetch_github_readme(repo: str) -> str:
    last_error: Exception | None = None
    for url in _github_readme_candidates(repo):
        try:
            _, body = _fetch_url(url)
            return body
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
    raise RuntimeError(f"could not fetch README for repo {repo}: {last_error}")


def _github_blob_to_raw(url: str) -> str:
    pattern = r"https://github\.com/([^/]+/[^/]+)/blob/([^/]+)/(.+)"
    match = re.match(pattern, url)
    if not match:
        return url
    repo, branch, path = match.groups()
    owner, name = repo.split("/", 1)
    return f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{path}"


def _parse_raw_github_url(url: str) -> tuple[str, str, str, str] | None:
    pattern = r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)"
    match = re.match(pattern, url)
    if not match:
        return None
    owner, name, branch, path = match.groups()
    return owner, name, branch, path


def _extract_markdown_links(markdown: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", markdown):
        target = match.group(1).strip().strip("<>").strip()
        if not target or target.startswith("#") or target.startswith("mailto:"):
            continue
        links.append(target.split("#", 1)[0])
    return links


def _resolve_markdown_link(base_url: str, link: str) -> tuple[str, str] | None:
    link = link.strip()
    if not link:
        return None
    if re.match(r"https?://", link):
        raw_url = _github_blob_to_raw(link)
        parsed = _parse_raw_github_url(raw_url)
        if parsed is None:
            return None
        owner, name, branch, path = parsed
        return raw_url, path
    parsed_base = _parse_raw_github_url(base_url)
    if parsed_base is None:
        return None
    owner, name, branch, base_path = parsed_base
    base_dir = posixpath.dirname(base_path)
    resolved_path = posixpath.normpath(posixpath.join(base_dir, link))
    raw_url = f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{resolved_path}"
    return raw_url, resolved_path


def _fetch_linked_markdown_readmes(source: KnowledgeSourceConfig) -> tuple[str, str]:
    base_url = _github_blob_to_raw(source.url or "")
    _, markdown = _fetch_url(base_url)
    filter_prefix = (source.path or "").strip()
    fetched_sections: list[str] = []
    seen_paths: set[str] = set()
    for link in _extract_markdown_links(markdown):
        resolved = _resolve_markdown_link(base_url, link)
        if resolved is None:
            continue
        raw_url, rel_path = resolved
        if not rel_path.lower().endswith(".md"):
            continue
        if filter_prefix and not rel_path.startswith(filter_prefix):
            continue
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        try:
            _, body = _fetch_url(raw_url)
        except Exception as exc:  # pragma: no cover - network path
            fetched_sections.append(f"## {rel_path}\nFetch failed: {exc}")
            if len(fetched_sections) >= source.max_results:
                break
            continue
        fetched_sections.append(f"## {rel_path}\n{body}")
        if len(fetched_sections) >= source.max_results:
            break
    if not fetched_sections:
        raise RuntimeError(f"no linked markdown files matched filter {filter_prefix!r} from {base_url}")
    preamble = "Fetched linked markdown files:\n" + "\n".join(
        f"- {section.splitlines()[0][3:]}" for section in fetched_sections if section.startswith("## ")
    )
    return base_url, preamble + "\n\n" + "\n\n".join(fetched_sections)


def _fetch_text_source(source: KnowledgeSourceConfig) -> tuple[str, str]:
    if source.kind == "url":
        url = source.url or ""
        content_type, body = _fetch_url(_github_blob_to_raw(url))
        if "html" in content_type.lower():
            parser = _HTMLTextExtractor()
            parser.feed(body)
            return url, parser.get_text()
        return url, body
    if source.kind == "github_readme":
        repo = source.repo or ""
        return f"https://github.com/{repo}", _fetch_github_readme(repo)
    if source.kind == "github_markdown_linked_readmes":
        return _fetch_linked_markdown_readmes(source)
    if source.kind == "arxiv_search":
        query = urllib.parse.quote(source.query or "")
        url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={source.max_results}"
        _, body = _fetch_url(url)
        root = ET.fromstring(body)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
            link = ""
            for elem in entry.findall("atom:link", ns):
                href = elem.attrib.get("href", "")
                rel = elem.attrib.get("rel", "")
                if href and rel == "alternate":
                    link = href
                    break
            entries.append(f"Title: {title}\nLink: {link}\nSummary: {summary}")
        return url, "\n\n".join(entries)
    raise RuntimeError(f"unsupported knowledge source kind: {source.kind}")


KNOWLEDGE_SYSTEM_PROMPT = """You are summarizing external technical sources into a compact research memory card.
Return JSON only.

Focus on:
- facts or techniques relevant to the current problem
- practical lessons
- cautions or failure modes
- concrete search directions suggested by the source
"""


def _build_summary_prompt(config: CampaignConfig, source: KnowledgeSourceConfig, source_url: str, content: str) -> str:
    trimmed = content[: source.max_chars]
    return f"""Problem:
{config.problem_statement}

Source id:
{source.id}

Source kind:
{source.kind}

Source URL:
{source_url}

Return JSON with:
{{
  "title": "short card title",
  "summary": "2-4 sentence summary",
  "key_points": ["important point", "..."],
  "tactical_ideas": ["experiment idea", "..."],
  "cautions": ["risk or warning", "..."],
  "tags": ["short-tag", "..."]
}}

Source content:
{trimmed}
"""


class KnowledgeManager:
    def __init__(self, campaign_dir: Path, config: CampaignConfig, llm: BaseLLM):
        self.root = campaign_dir / "knowledge"
        self.cards_path = self.root / "cards.json"
        self.config = config
        self.llm = llm

    def load_cards(self) -> list[dict[str, Any]]:
        return _read_json(self.cards_path, [])

    def refresh(self, force: bool = False) -> list[dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        existing = {item.get("source_id"): item for item in self.load_cards()}
        cards: list[dict[str, Any]] = []
        for source in self.config.knowledge_sources:
            old = existing.get(source.id)
            if old and not force and not self._is_stale(old, source.refresh_seconds):
                cards.append(old)
                continue
            try:
                source_url, content = _fetch_text_source(source)
                card = self._summarize_source(source, source_url, content)
            except Exception as exc:
                card = {
                    "source_id": source.id,
                    "source_kind": source.kind,
                    "source_url": source.url or source.repo or source.query or "",
                    "fetched_at": _now_iso(),
                    "title": f"{source.id} (fetch failed)",
                    "summary": f"Knowledge fetch failed: {exc}",
                    "key_points": [],
                    "tactical_ideas": [],
                    "cautions": [str(exc)],
                    "tags": ["fetch-failed"],
                }
            cards.append(card)
        _write_json(self.cards_path, cards)
        return cards

    def format_cards(self, limit: int = 8) -> str:
        cards = self.load_cards()[:limit]
        if not cards:
            return "None."
        lines: list[str] = []
        for card in cards:
            lines.append(
                "- {title} [{source_id}]: {summary}\n  key_points={kp}\n  tactical_ideas={ideas}\n  cautions={cautions}".format(
                    title=card.get("title", ""),
                    source_id=card.get("source_id", ""),
                    summary=card.get("summary", ""),
                    kp="; ".join(card.get("key_points", [])[:3]),
                    ideas="; ".join(card.get("tactical_ideas", [])[:3]),
                    cautions="; ".join(card.get("cautions", [])[:2]),
                )
            )
        return "\n".join(lines)

    def _is_stale(self, card: dict[str, Any], refresh_seconds: int) -> bool:
        fetched_at = card.get("fetched_at")
        if not fetched_at:
            return True
        try:
            then = datetime.fromisoformat(str(fetched_at))
        except ValueError:
            return True
        age = datetime.now(UTC) - then.astimezone(UTC)
        return age.total_seconds() >= refresh_seconds

    def _summarize_source(self, source: KnowledgeSourceConfig, source_url: str, content: str) -> dict[str, Any]:
        prompt = _build_summary_prompt(self.config, source, source_url, content)
        try:
            summary = self.llm.generate_json("knowledge_summary", KNOWLEDGE_SYSTEM_PROMPT, prompt)
        except LLMError:
            summary = _heuristic_summary(source, content)
        return {
            "source_id": source.id,
            "source_kind": source.kind,
            "source_url": source_url,
            "fetched_at": _now_iso(),
            "title": str(summary.get("title", source.id)),
            "summary": str(summary.get("summary", "")),
            "key_points": [str(item) for item in summary.get("key_points", [])],
            "tactical_ideas": [str(item) for item in summary.get("tactical_ideas", [])],
            "cautions": [str(item) for item in summary.get("cautions", [])],
            "tags": [str(item) for item in summary.get("tags", [])],
        }
