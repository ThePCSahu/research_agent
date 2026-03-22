"""
Analyzer — LLM-backed extraction of insights, gaps, and confidence from text chunks.

Fits the pipeline: … Retrieve → **Analyze** → Decide → Loop …
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from research_agent.models.llm_client import LLMClient


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research analyst. You receive excerpts (chunks) from retrieved documents \
about a topic.

Your tasks:
1. **Summarize key findings** — concrete, factual takeaways supported by the text. \
Each insight should be one short sentence or phrase.
2. **Identify missing areas** — important angles, data, or questions the chunks do \
**not** adequately cover (gaps for further search or reading).
3. **Assign confidence** — a number from **0.0** to **1.0** reflecting how well \
the combined evidence supports solid conclusions (coverage, consistency, recency implied \
in the text). Use **0.0** if the chunks are empty or uninformative.

Rules:
- Base insights and gaps only on what is reasonable to infer from the given chunks; \
mark gaps explicitly when evidence is thin.
- Return **only** a single JSON object with exactly these keys:
  - "insights": array of strings
  - "gaps": array of strings
  - "confidence": number (float between 0.0 and 1.0)
- No markdown fences, no commentary outside the JSON.
"""


def _format_chunks(chunks: List[str]) -> str:
    parts = [c.strip() for c in chunks if c and str(c).strip()]
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)


def _clamp_confidence(value: Any) -> float:
    """Map model output to [0.0, 1.0]; values in (1, 100] are treated as percentages."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if x < 0.0:
        return 0.0
    if x <= 1.0:
        return round(x, 4)
    if x <= 100.0:
        return round(min(x / 100.0, 1.0), 4)
    return 1.0


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _parse_analysis(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning("Analyzer JSON parse failed — returning empty result")
    return None


def _normalize_payload(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"insights": [], "gaps": [], "confidence": 0.0}
    insights = _string_list(data.get("insights", []))
    gaps = _string_list(data.get("gaps", []))
    conf = _clamp_confidence(data.get("confidence", 0.0))
    return {"insights": insights, "gaps": gaps, "confidence": conf}


class DocumentAnalyzer:
    """Extract key insights, coverage gaps, and a confidence score from text chunks."""
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client or LLMClient()
        
    def analyze(self, chunks: list[str]) -> dict:
        """Extract key insights, coverage gaps, and a confidence score from *chunks*."""
        body = _format_chunks(chunks)
        if not body:
            logger.info("Analyzer: no chunk text — skipping LLM")
            return {"insights": [], "gaps": [], "confidence": 0.0}

        user_content = (
            "Below are text chunks from retrieved sources. Analyze them.\n\n" + body
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        raw = self.llm.chat(messages=messages, temperature=0.2)
        logger.debug("Analyzer raw LLM response:\n%s", raw)

        parsed = _parse_analysis(raw)
        result = _normalize_payload(parsed)
        logger.info(
            "Analyzer: %d insights, %d gaps, confidence=%s",
            len(result["insights"]),
            len(result["gaps"]),
            result["confidence"],
        )
        return result
