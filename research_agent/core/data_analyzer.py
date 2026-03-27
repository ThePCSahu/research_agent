"""
Data Analyzer — Combined extraction of insights, gaps, confidence, and next-step decisions.

Replaces the separate DocumentAnalyzer and DecisionEngine.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Dict
from research_agent.utils.config import get_config_or_default
from research_agent.models.llm_client import LLMClient
from research_agent.memory.state import AgentState
from .query_planner import QueryPlanner


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research data analyzer. You receive excerpts (chunks) from retrieved documents \
about the topic mentioned in the user message.

Your tasks:
1. **Summarize key findings** — concrete, factual takeaways supported by the text.
2. **Identify consensus & contradictions** — highlight where sources agree and where they conflict.
3. **Identify missing areas** — important angles, data, or questions the chunks do **not** adequately cover (gaps).
4. **Evaluate source credibility** — note if any sources seem particularly authoritative or potentially biased based on their content/excerpts.
5. **Assign confidence** — a number from **0.0** to **1.0**. Confidence reflects the **completeness and reliability** of the evidence. 
   - High confidence (>0.8) means you have enough consistent evidence to answer the topic thoroughly.
   - Identifying contradictions is a sign of thorough analysis and should NOT automatically lower confidence if the overall picture is clear.
   - If the "Document Chunks" (working memory) provide strong, cross-validated support for insights, confidence should be high.
6. **Propose next steps** — If significant gaps or *unresolved* contradictions remain, suggest **{min_q}-{max_q} highly targeted web-search queries **.

Return **only** a single JSON object with exactly these keys:
{
  "insights": [ string, ... ],
  "contradictions": [ string, ... ],
  "gaps": [ string, ... ],
  "confidence": number,
  "sources_evaluation": [ string, ... ],
  "queries": [ string, ... ]
}

Rules:
- Base results ONLY on the provided chunks.
- If confidence is high and no major gaps remain, "queries" can be [].
- No markdown fences, no commentary outside the JSON.
"""

def _format_chunks(chunks: List[str]) -> str:
    parts = [c.strip() for c in chunks if c and str(c).strip()]
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)

def _clamp_confidence(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, round(x, 4) if x <= 1.0 else round(x / 100.0, 4)))

def _parse_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

class DataAnalyzer:
    """Unified component for analyzing chunks and deciding next search steps."""

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client or LLMClient()
        self.min_q = get_config_or_default("AGENT_MIN_QUERIES", "2")
        self.max_q = get_config_or_default("AGENT_MAX_QUERIES", "4")

    def analyze(self, chunks: list[str], state: AgentState) -> dict:
        """Analyze chunks and return insights, gaps, confidence, and next queries."""
        body = _format_chunks(chunks)
        if not body:
            return {"insights": [], "gaps": [], "confidence": 0.0, "queries": []}

        user_content = (
            f"Topic: '{state.topic}'\n"
            f"Queries done: {list(state.queries_done)}\n\n"
            f"Document Chunks:\n{body}"
        )
        
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(min_q=self.min_q, max_q=self.max_q)},
            {"role": "user", "content": user_content},
        ]
        
        raw = self.llm.chat(messages=messages, temperature=0.2)
        logger.debug("DataAnalyzer raw output: %s", raw)
        
        parsed = _parse_json(raw) or {}
        
        # Keep queries as a list of dicts: [{"query": "...", "hyde": "..."}]
        return {
            "insights": [str(i) for i in parsed.get("insights", []) if str(i).strip()],
            "contradictions": [str(c) for c in parsed.get("contradictions", []) if str(c).strip()],
            "gaps": [str(g) for g in parsed.get("gaps", []) if str(g).strip()],
            "confidence": _clamp_confidence(parsed.get("confidence", 0.0)),
            "sources_evaluation": [str(s) for s in parsed.get("sources_evaluation", []) if str(s).strip()],
            "queries": parsed.get("queries", []),
        }
