"""Tests for research_agent.core.document_analyzer."""

from __future__ import annotations

import json
from unittest.mock import patch

from research_agent.core.document_analyzer import (
    DocumentAnalyzer,
    _clamp_confidence,
    _normalize_payload,
    _parse_analysis,
)


def test_clamp_confidence_in_range():
    assert _clamp_confidence(0.5) == 0.5
    assert _clamp_confidence(0) == 0.0
    assert _clamp_confidence(1) == 1.0
    assert _clamp_confidence(-0.1) == 0.0
    assert _clamp_confidence(150) == 1.0


def test_clamp_confidence_percent_scale():
    assert _clamp_confidence(75) == 0.75
    assert _clamp_confidence(2) == 0.02


def test_clamp_confidence_invalid():
    assert _clamp_confidence("nope") == 0.0


def test_normalize_payload_full():
    out = _normalize_payload(
        {
            "insights": ["  a ", "b"],
            "gaps": ["need more data"],
            "confidence": 0.82,
        }
    )
    assert out == {
        "insights": ["a", "b"],
        "gaps": ["need more data"],
        "confidence": 0.82,
    }


def test_normalize_payload_missing_keys():
    out = _normalize_payload({})
    assert out == {"insights": [], "gaps": [], "confidence": 0.0}


def test_parse_analysis_raw_json():
    raw = '{"insights":["x"],"gaps":["y"],"confidence":0.4}'
    assert _parse_analysis(raw) == {
        "insights": ["x"],
        "gaps": ["y"],
        "confidence": 0.4,
    }


def test_parse_analysis_embedded_object():
    raw = 'Here you go:\n{"insights": [], "gaps": ["z"], "confidence": 0.1}\n'
    d = _parse_analysis(raw)
    assert d is not None
    assert d["gaps"] == ["z"]


def test_analyze_empty_chunks_skips_llm():
    analyzer = DocumentAnalyzer()
    with patch.object(analyzer.llm, "chat") as mock_chat:
        out = analyzer.analyze([])
        mock_chat.assert_not_called()
    assert out == {"insights": [], "gaps": [], "confidence": 0.0}

    with patch.object(analyzer.llm, "chat") as mock_chat:
        out = analyzer.analyze(["", "  "])
        mock_chat.assert_not_called()
    assert out == {"insights": [], "gaps": [], "confidence": 0.0}


def test_analyze_calls_llm_and_returns_parsed_dict(mock_llm_client):
    payload = {
        "insights": ["Finding one", "Finding two"],
        "gaps": ["Regional data missing"],
        "confidence": 0.65,
    }
    mock_llm_client.chat.return_value = json.dumps(payload)

    analyzer = DocumentAnalyzer(llm_client=mock_llm_client)
    out = analyzer.analyze(["Chunk about topic A.", "Chunk about topic B."])

    mock_llm_client.chat.assert_called_once()
    assert out["insights"] == ["Finding one", "Finding two"]
    assert out["gaps"] == ["Regional data missing"]
    assert out["confidence"] == 0.65
