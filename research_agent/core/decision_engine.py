"""
Decision engine — LLM-guided control for the research loop.

Fits the pipeline: … Retrieve → Analyze → **Decide** → Loop …
"""

from __future__ import annotations

import os
import json
import logging
import re
from typing import Any, List

from research_agent.models.llm_client import LLMClient
from research_agent.memory.state import AgentState
from research_agent.utils.config import get_config_or_default
from .query_planner import QueryPlanner


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the control module for a research agent. The message lists **gaps** \
(what is still missing), **insights** so far, **queries already run**, and the \
**iteration** count.

Propose **new web-search queries** that directly target the gaps. Do **not** repeat \
or trivially rephrase queries already listed under queries done.

Return **only** a JSON object:
{ "action": "continue", "queries": [ string, ... ] }

Rules:
- Always set **"action"** to **"continue"** here (the runtime will stop the loop \
when there are no gaps).
- **"queries"** must be a non-empty list of 3–8 distinct, Google-style queries \
(5–15 words each) unless you truly cannot suggest any new angle — then use [].
"""


def _normalize_query(q: str) -> str:
    return " ".join(q.split()).strip().lower()


def _normalized_gaps(state: AgentState) -> List[str]:
    return [g.strip() for g in state.gaps if g and str(g).strip()]


def _format_state_for_llm(state: AgentState) -> str:
    gaps = _normalized_gaps(state)
    insights = [i.strip() for i in state.insights if i and str(i).strip()]
    done = list(state.queries_done)
    lines = [
        f"iteration: {state.iteration}",
        f"gaps ({len(gaps)}):",
    ]
    lines.extend(f"  - {g}" for g in gaps)
    lines.append(f"insights ({len(insights)}):")
    lines.extend(f"  - {x}" for x in insights[:20])
    if len(insights) > 20:
        lines.append("  …")
    lines.append(f"queries already done ({len(done)}):")
    lines.extend(f"  - {q}" for q in done[:30])
    if len(done) > 30:
        lines.append("  …")
    return "\n".join(lines)


def _parse_decision(raw: str) -> dict[str, Any] | None:
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
    return None


def _dedupe_queries_against_state(queries: List[str], state: AgentState) -> List[str]:
    done_keys = {_normalize_query(q) for q in state.queries_done}
    out: List[str] = []
    seen: set[str] = set()
    for q in queries:
        key = _normalize_query(q)
        if not key or key in done_keys or key in seen:
            continue
        seen.add(key)
        out.append(" ".join(q.split()).strip())
    return out


class DecisionEngine:
    """Return the next loop action and optional search queries."""

    def __init__(self, llm_client: Any = None, query_planner: Any = None):
        self.llm = llm_client or LLMClient()
        self.query_planner = query_planner or QueryPlanner(llm_client=self.llm)

    def decide_next_step(self, state: AgentState) -> dict:
        """Return the next loop action and optional search queries.

        * No gaps → ``{"action": "finish", "queries": []}`` (no LLM).
        * Gaps present → LLM proposes queries; falls back to :func:`generate_queries` \
          if needed; respects ``MAX_ITERATIONS``.
        """

        max_iterations = int(get_config_or_default("AGENT_MAX_ITERATIONS", "5"))
        if state.iteration >= max_iterations:
            logger.info("Decision: max iterations (%s) — finish", max_iterations)
            return {"action": "finish", "queries": []}

        gaps = _normalized_gaps(state)
        if not gaps:
            logger.info("Decision: no gaps — finish")
            return {"action": "finish", "queries": []}

        user_content = (
            "Decide the next step given the following state.\n\n"
            + _format_state_for_llm(state)
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        raw = self.llm.chat(messages=messages, temperature=0.3)
        logger.debug("Decision raw LLM response:\n%s", raw)

        parsed = _parse_decision(raw) or {}
        raw_queries = parsed.get("queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = []
        queries = [
            " ".join(str(q).split()).strip()
            for q in raw_queries
            if str(q).strip()
        ]
        queries = _dedupe_queries_against_state(queries, state)

        if not queries:
            logger.info("Decision: empty queries after LLM — fallback to query generator from gaps")
            topic = "; ".join(gaps[:12])
            queries = self.query_planner.generate_queries_from_gaps(topic)
            queries = _dedupe_queries_against_state(queries, state)

        if not queries:
            logger.info("Decision: no new queries available — finish")
            return {"action": "finish", "queries": []}

        return {"action": "continue", "queries": queries}
