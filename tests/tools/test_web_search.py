import os
import pytest
from unittest.mock import patch, MagicMock
from research_agent.tools.web_search import web_search, WEB_SEARCH_TOOL


def test_web_search_success(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test_key")

    mock_response_data = {
        "organic_results": [
            {
                "title": "Quantum Computing",
                "link": "https://example.com/qc",
                "snippet": "A short summary of qc."
            },
            {
                "title": "Qubits",
                "link": "https://example.com/qubits",
                "snippet": "Explain qubits."
            }
        ]
    }

    with patch('research_agent.tools.web_search.requests.post') as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status.return_value = None
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        output = web_search("quantum", top_k=2)

        mock_post.assert_called_once()

        assert len(output) == 2

        assert output[0]["title"] == "Quantum Computing"
        assert output[0]["url"] == "https://example.com/qc"
        assert output[0]["body"] == "A short summary of qc."

        assert output[1]["title"] == "Qubits"
        assert output[1]["url"] == "https://example.com/qubits"
        assert output[1]["body"] == "Explain qubits."


def test_web_search_empty(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test_key")

    with patch('research_agent.tools.web_search.requests.post') as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"organic_results": []}
        mock_resp.raise_for_status.return_value = None
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        output = web_search("quantum", top_k=2)
        assert output == []


def test_web_search_missing_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    output = web_search("quantum", top_k=2)
    assert len(output) == 1
    assert output[0]["title"] == "Error"


def test_web_search_error(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test_key")

    with patch('research_agent.tools.web_search.requests.post') as mock_post:
        mock_post.side_effect = Exception("API limit exceeded")

        output = web_search("quantum", top_k=2)
        assert len(output) == 1
        assert output[0]["title"] == "Error"
        assert "API limit exceeded" in output[0]["body"]


def test_tool_definition():
    assert WEB_SEARCH_TOOL.name == "web_search"
    assert "query" in WEB_SEARCH_TOOL.parameters["required"]
