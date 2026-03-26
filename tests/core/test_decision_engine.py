"""Tests for research_agent.core.decision_engine."""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

from research_agent.core.decision_engine import (
    DecisionEngine,
    _dedupe_queries_against_state,
    _format_state_for_llm,
    _normalized_gaps,
)
from research_agent.memory.state import AgentState


def test_no_gaps_returns_finish_without_llm():
    state = AgentState(gaps=[], insights=["i1"], queries_done=["q1"])
    engine = DecisionEngine()
    with patch.object(engine.llm, "chat") as mock_chat:
        out = engine.decide_next_step(state)
        mock_chat.assert_not_called()
    assert out == {"action": "finish", "queries": []}


def test_whitespace_only_gaps_treated_as_no_gaps():
    state = AgentState(gaps=["", "  \t"])
    engine = DecisionEngine()
    with patch.object(engine.llm, "chat") as mock_chat:
        out = engine.decide_next_step(state)
        mock_chat.assert_not_called()
    assert out["action"] == "finish"


def test_max_iterations_returns_finish_without_llm():
    state = AgentState(
        gaps=["still missing X"],
        iteration=5,
    )
    with patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "5"}):
        engine = DecisionEngine()
        with patch.object(engine.llm, "chat") as mock_chat:
            out = engine.decide_next_step(state)
            mock_chat.assert_not_called()
    assert out == {"action": "finish", "queries": []}


def test_gaps_triggers_llm_and_returns_continue(mock_llm_client):
    payload = {
        "action": "continue",
        "queries": [
            "gap-specific search one",
            "gap-specific search two",
        ],
    }
    mock_llm_client.chat.return_value = json.dumps(payload)
    state = AgentState(gaps=["need more on pricing"], queries_done=[])

    engine = DecisionEngine(llm_client=mock_llm_client)
    out = engine.decide_next_step(state)

    mock_llm_client.chat.assert_called_once()
    assert out["action"] == "continue"
    assert out["queries"] == payload["queries"]


def test_fallback_to_planner_when_llm_returns_no_queries(mock_llm_client):
    mock_llm_client.chat.return_value = json.dumps({"action": "continue", "queries": []})
    state = AgentState(gaps=["regional adoption data"])

    planned = ["planner q1", "planner q2"]
    
    # Mock query generator
    mock_qg = MagicMock()
    mock_qg.generate_queries_from_gaps.return_value = planned

    engine = DecisionEngine(llm_client=mock_llm_client, query_planner=mock_qg)
    out = engine.decide_next_step(state)

    mock_qg.generate_queries_from_gaps.assert_called_once()
    assert out == {"action": "continue", "queries": planned}


def test_dedupe_queries_against_state():
    state = AgentState(queries_done=["already ran"])
    out = _dedupe_queries_against_state(
        ["already ran", "  ALREADY RAN  ", "new angle"],
        state,
    )
    assert out == ["new angle"]


def test_normalized_gaps_and_format_includes_iteration():
    state = AgentState(
        gaps=[" a ", "b"],
        insights=["i"],
        queries_done=["q"],
        iteration=2,
    )
    assert _normalized_gaps(state) == ["a", "b"]
    text = _format_state_for_llm(state)
    assert "iteration: 2" in text
    assert "a" in text and "b" in text
