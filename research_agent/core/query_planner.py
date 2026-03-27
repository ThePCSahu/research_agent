"""
Query Planner — decompose a topic into targeted web-search queries.

Uses :class:`~research_agent.llm_client.LLMClient` to analyse a research topic and
produce multiple diverse queries that cover different facets of the subject.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from research_agent.models.llm_client import LLMClient
from research_agent.utils.config import get_config_or_default


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research strategist. Your job is to take the broad research topic mentioned in the user message \
and break it down into **{min_q}-{max_q} highly targeted web-search queries**.

For EACH query, you must also provide a short (1-2 paragraph) hypothetical but highly accurate \
and detailed response that answers the query (HyDE). This response should contain technical terms \
and facts expected in an authoritative source.

Rules:
1. Each query must focus on a DIFFERENT angle of the topic.
2. Return ONLY a JSON array of objects.
3. Each object must have "query" and "hyde" keys.

[
  {{"query": "q1", "hyde": "hypothetical answer 1"}},
  {{"query": "q2", "hyde": "hypothetical answer 2"}}
]
"""


class QueryPlanner:
    """Analyse a topic and produce smart, multi-angle search queries."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm = llm_client or LLMClient()
        self.min_q = get_config_or_default("AGENT_MIN_QUERIES", "2")
        self.max_q = get_config_or_default("AGENT_MAX_QUERIES", "4")

    def generate_queries(self, topic: str) -> List[Dict[str, str]]:
        """Return a list of targeted search queries and their HyDE answers."""
        logger.info("Generating search queries/hyde for topic: %s", topic)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(min_q=self.min_q, max_q=self.max_q)},
            {"role": "user", "content": f"Generate queries and hyde for the topic: '{topic}'"},
        ]

        raw = self.llm.chat(messages=messages, temperature=0.7)
        return self._parse_queries(raw)

    @staticmethod
    def _parse_queries(raw: str) -> List[Dict[str, str]]:
        """Parse the LLM output into a list of {query, hyde} objects."""
        try:
            import json
            import re
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [q for q in parsed if isinstance(q, dict) and "query" in q and "hyde" in q]
        except Exception as e:
            logger.error(f"Failed to parse queries: {e}")
        
        return []

    @staticmethod
    def _normalize_query(q: str) -> str:
        return " ".join(q.split()).strip()

    @classmethod
    def _dedupe_queries(cls, queries: List[str]) -> List[str]:
        """Preserve order; drop exact duplicates after whitespace normalization (case-insensitive)."""
        seen: set[str] = set()
        out: List[str] = []
        for q in queries:
            key = cls._normalize_query(q).lower()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(cls._normalize_query(q))
        return out
