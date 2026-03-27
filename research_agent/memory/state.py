"""In-memory agent progress tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set


def _normalize_query(q: str) -> str:
    return " ".join(q.split()).strip()


@dataclass
class AgentState:
    """Tracks research-agent progress across iterations."""

    topic: str
    queries_done: List[str] = field(default_factory=list)
    urls_fetched: Set[str] = field(default_factory=set)
    insights: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    sources_evaluation: List[str] = field(default_factory=list)
    working_memory: List[dict] = field(default_factory=list)
    iteration: int = 0
    _queries_seen: Set[str] = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        seen: Set[str] = set()
        deduped: List[str] = []
        for q in self.queries_done:
            key = _normalize_query(q)
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)
        self.queries_done = deduped
        self._queries_seen = seen

    def record_query(self, query: str) -> bool:
        """Register a completed query; skips duplicates (whitespace-normalized).

        Returns True if the query was new and appended.
        """
        key = _normalize_query(query)
        if not key:
            return False
        if key in self._queries_seen:
            return False
        self._queries_seen.add(key)
        self.queries_done.append(key)
        return True

    def record_url(self, url: str) -> bool:
        """Register a fetched URL; skips duplicates. Returns True if new."""
        u = url.strip()
        if not u:
            return False
        before = len(self.urls_fetched)
        self.urls_fetched.add(u)
        return len(self.urls_fetched) > before
