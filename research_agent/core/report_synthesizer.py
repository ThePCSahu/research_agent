"""
Synthesizer — LLM-backed Markdown report from retrieved chunks and metadata.

Fits the pipeline: Retrieve → **Synthesize** → Report.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from research_agent.models.llm_client import LLMClient


logger = logging.getLogger(__name__)

_MAX_USER_MESSAGE_CHARS = 48_000

_SYSTEM_PROMPT = """\
You are the final synthesizer for a research deliverable about the topic mentioned in the user message.

You receive numbered **source excerpts**, a **reference list** with titles and URLs, and an **analysis summary** containing insights, contradictions, and source evaluations.

Produce one cohesive **Markdown** document.

Requirements:
1. **Executive Summary** — Start with a brief overview of the key insights.
2. **Consensus & Contradictions** — Group content by theme. Explicitly address areas where sources agree and where they conflict. Use `##` for major themes and `###` for subtopics.
3. **Cite sources** using the bracket numbers from the reference list, e.g. [2] or [1][3].
4. **Source Reliability** — Briefly mention the credibility or limitations of the sources used if relevant (based on the provided evaluation).
5. **## Sources** section — numbered lines matching the reference list: `[n] **Title** — URL`.
6. Write in clear, neutral prose. Do not invent facts.
7. Return **only** the Markdown report.
"""


def _normalize_chunk(d: dict) -> Tuple[str, Dict[str, Any]]:
    text = str(d.get("text") or d.get("content") or "").strip()
    raw_meta = d.get("metadata")
    meta: Dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    for key in ("url", "title", "source", "published_at", "author"):
        if key in d and d[key] is not None and key not in meta:
            meta[key] = d[key]
    return text, meta


def _reference_line(n: int, meta: Dict[str, Any]) -> str:
    title = str(meta.get("title") or meta.get("source") or "").strip() or "(untitled)"
    url = str(meta.get("url") or "").strip()
    extra = f" — {url}" if url else ""
    return f"[{n}] {title}{extra}"


def _format_chunk_bundle(chunks: List[dict]) -> str:
    """Build user message: reference block + excerpt block."""
    lines_ref: List[str] = ["## Reference metadata (use these numbers for citations in the report)"]
    lines_ex: List[str] = ["## Excerpts (synthesize only from this text)"]

    n = 0
    for d in chunks:
        text, meta = _normalize_chunk(d)
        if not text:
            continue
        n += 1
        lines_ref.append(_reference_line(n, meta))
        lines_ex.append(f"### Excerpt [{n}]")
        lines_ex.append(text)

    if n == 0:
        return ""

    return "\n".join(lines_ref) + "\n\n" + "\n".join(lines_ex)


def _strip_wrapping_fences(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw


class ReportSynthesizer:
    """Synthesizer — LLM-backed Markdown report generator from text chunks."""

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client or LLMClient()

    def generate_report(self, chunks: list[dict], state: AgentState) -> str:
        """Return a Markdown report for *topic* grouped by theme with citations from chunk metadata."""
        bundle = _format_chunk_bundle(chunks)
        if not bundle:
            logger.info("Synthesizer: no chunk text — returning stub report")
            return "## Report\n\n*No source excerpts were provided.*\n"

        if len(bundle) > _MAX_USER_MESSAGE_CHARS:
            bundle = bundle[:_MAX_USER_MESSAGE_CHARS] + "\n\n… *(truncated)*\n"
            logger.warning("Synthesizer: user message truncated to %s chars", _MAX_USER_MESSAGE_CHARS)

        analysis_summary = (
            f"Key Insights: {json.dumps(state.insights, indent=2)}\n"
            f"Contradictions: {json.dumps(state.contradictions, indent=2)}\n"
            f"Source Evaluations: {json.dumps(state.sources_evaluation, indent=2)}\n"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Produce the Markdown report for the topic: '{state.topic}'\n\n"
                    f"--- Analysis Summary ---\n{analysis_summary}\n\n"
                    f"--- Source Data ---\n{bundle}"
                ),
            },
        ]
        raw = self.llm.chat(messages=messages, temperature=0.25)
        logger.debug("Synthesizer raw LLM response (truncated): %s...", raw[:500])
        report = _strip_wrapping_fences(raw)
        if not report.strip():
            return "## Report\n\n*The model returned empty content.*\n"
        return report
