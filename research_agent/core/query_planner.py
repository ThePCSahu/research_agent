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


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research strategist. Your job is to take a broad research topic \
and break it down into **5 to 8 highly targeted web-search queries**.

Rules:
1. Each query must focus on a DIFFERENT angle of the topic.
   Cover a mix of these perspectives where relevant:
   - Definitions & fundamentals
   - Recent news & developments (include the current year when helpful)
   - Expert opinions & academic research
   - Real-world applications & case studies
   - Statistics, data & market trends
   - Comparisons & alternatives
   - Challenges, risks & criticisms
   - Future outlook & predictions
2. Queries should be concise (5-15 words) and phrased exactly as someone \
   would type them into Google.
3. Do NOT repeat the same angle twice.
4. Return ONLY a JSON array of strings — no explanation, no markdown, \
   no numbering.

["query one", "query two"]
"""

_GAPS_SYSTEM_PROMPT = """\
You are a research strategist. Given a topic, output **5 to 8 distinct web-search \
queries** a researcher would run in Google.

Each query must target a **different angle**. Cover these facets where they apply \
to the topic (skip only what is clearly irrelevant):

1. **Overview** — definitions, scope, background, how it works
2. **Statistics** — numbers, data, benchmarks, measurable outcomes
3. **Trends** — recent changes, forecasts, adoption, momentum over time
4. **Geography** — regional or country-specific angles, comparisons across places
5. **Edge cases** — limits, failures, rare scenarios, controversies, risks

Rules:
- Queries: 5–15 words each, natural search-engine phrasing.
- Do **not** repeat the same angle or near-duplicate wording.
- Return **only** a JSON array of strings — no markdown, no commentary.

Example:
["query one", "query two"]
"""


class QueryPlanner:
    """Analyse a topic and produce smart, multi-angle search queries."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm = llm_client or LLMClient()

    def generate_queries(self, topic: str) -> List[str]:
        """Return a list of targeted search queries for *topic*.

        Parameters
        ----------
        topic : str
            The research topic to decompose.

        Returns
        -------
        list[str]
            5–8 search-engine-ready query strings.
        """
        logger.info("Generating search queries for topic: %s", topic)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": topic},
        ]

        raw = self.llm.chat(messages=messages, temperature=0.7)
        logger.debug("Raw LLM response:\n%s", raw)

        queries = self._parse_queries(raw)
        logger.info("Generated %d queries for topic '%s'", len(queries), topic)
        return queries

    def generate_queries_from_gaps(self, topic: str) -> List[str]:
        """Use the LLM to produce diverse, non-overlapping search queries for *topic*."""
        logger.info("Planner generating queries for topic: %s", topic)

        messages = [
            {"role": "system", "content": _GAPS_SYSTEM_PROMPT},
            {"role": "user", "content": topic},
        ]
        raw = self.llm.chat(messages=messages, temperature=0.7)
        logger.debug("Planner raw LLM response:\n%s", raw)

        parsed = self._parse_queries(raw)
        queries = self._dedupe_queries(parsed)
        logger.info("Planner produced %d queries after deduplication", len(queries))
        return queries


    @staticmethod
    def _parse_queries(raw: str) -> List[str]:
        """Parse the LLM output into a list of query strings.

        Tries JSON first, then falls back to a line-by-line heuristic
        (useful for smaller models that don't always follow JSON format).
        """
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return [q.strip() for q in parsed if q.strip()]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [str(q).strip() for q in parsed if str(q).strip()]
            except json.JSONDecodeError:
                pass

        logger.warning("JSON parsing failed — falling back to line splitting")
        queries: List[str] = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip("\"'")
            if cleaned:
                queries.append(cleaned)
        return queries

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

