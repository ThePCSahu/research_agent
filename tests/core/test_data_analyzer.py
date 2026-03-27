import pytest
import json
from unittest.mock import MagicMock, patch
from research_agent.core.data_analyzer import DataAnalyzer
from research_agent.memory.state import AgentState

@pytest.fixture
def mock_llm_client():
    return MagicMock()

def test_data_analyzer_analyze_basic(mock_llm_client):
    payload = {
        "insights": ["Insight 1"],
        "contradictions": ["Contradiction 1"],
        "gaps": ["Gap 1"],
        "confidence": 0.5,
        "sources_evaluation": ["Source 1 is good"],
        "queries": ["Query 1"]
    }
    mock_llm_client.chat.return_value = json.dumps(payload)
    
    analyzer = DataAnalyzer(llm_client=mock_llm_client)
    state = AgentState(topic="test topic")
    result = analyzer.analyze(["chunk 1"], state)
    
    assert result["insights"] == ["Insight 1"]
    assert result["contradictions"] == ["Contradiction 1"]
    assert result["gaps"] == ["Gap 1"]
    assert result["confidence"] == 0.5
    assert result["sources_evaluation"] == ["Source 1 is good"]
    assert result["queries"] == ["Query 1"]
    mock_llm_client.chat.assert_called_once()

def test_data_analyzer_empty_chunks():
    analyzer = DataAnalyzer()
    state = AgentState(topic="test")
    result = analyzer.analyze([], state)
    assert result["confidence"] == 0.0
    assert result["insights"] == []
    assert result["queries"] == []

def test_data_analyzer_malformed_json(mock_llm_client):
    mock_llm_client.chat.return_value = "Not JSON at all"
    analyzer = DataAnalyzer(llm_client=mock_llm_client)
    state = AgentState(topic="test")
    result = analyzer.analyze(["some text"], state)
    assert result["confidence"] == 0.0
    assert isinstance(result["insights"], list)
