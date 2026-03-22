"""Tests for research_agent.memory.state.AgentState."""

from __future__ import annotations

from research_agent.memory.state import AgentState


def test_record_query_appends_once_and_skips_duplicate():
    state = AgentState()
    assert state.record_query("climate policy") is True
    assert state.record_query("climate policy") is False
    assert state.queries_done == ["climate policy"]


def test_record_query_treats_whitespace_variants_as_same_query():
    state = AgentState()
    assert state.record_query("  foo   bar  ") is True
    assert state.record_query("foo bar") is False
    assert state.queries_done == ["foo bar"]


def test_record_query_empty_or_whitespace_only_returns_false():
    state = AgentState()
    assert state.record_query("") is False
    assert state.record_query("   \t  ") is False
    assert state.queries_done == []


def test_initial_queries_deduplicated_and_normalized():
    state = AgentState(
        queries_done=["a", "  a  ", "b", "b"],
    )
    assert state.queries_done == ["a", "b"]


def test_record_url_adds_once_and_skips_duplicate():
    state = AgentState()
    assert state.record_url("https://example.com/page") is True
    assert state.record_url("https://example.com/page") is False
    assert state.urls_fetched == {"https://example.com/page"}


def test_record_url_strips_whitespace_same_url():
    state = AgentState()
    assert state.record_url("  https://example.com  ") is True
    assert state.record_url("https://example.com") is False


def test_record_url_empty_returns_false():
    state = AgentState()
    assert state.record_url("") is False
    assert state.record_url("   ") is False
    assert state.urls_fetched == set()


def test_insights_gaps_and_iteration_are_independent_lists():
    state = AgentState()
    state.insights.append("insight one")
    state.gaps.append("gap one")
    state.iteration = 3
    assert state.insights == ["insight one"]
    assert state.gaps == ["gap one"]
    assert state.iteration == 3
