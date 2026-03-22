from unittest.mock import patch, MagicMock
from research_agent.tools.web_search import web_search, WEB_SEARCH_TOOL

def test_web_search_success():
    mock_results = [
        {
            "title": "Quantum Computing",
            "href": "https://example.com/qc",
            "body": "A short summary of qc."
        },
        {
            "title": "Qubits",
            "href": "https://example.com/qubits",
            "body": "Explain qubits."
        }
    ]

    with patch('research_agent.tools.web_search.DDGS') as MockDDGS:
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_results
        MockDDGS.return_value = mock_instance

        output = web_search("quantum", top_k=2)
        
        # Verify it hit DDGS.text
        mock_instance.text.assert_called_once_with("quantum", max_results=4)

        # Verify formatting
        assert len(output) == 2
        
        assert output[0]["title"] == "Quantum Computing"
        assert output[0]["url"] == "https://example.com/qc"
        assert output[0]["body"] == "A short summary of qc."
        
        assert output[1]["title"] == "Qubits"
        assert output[1]["url"] == "https://example.com/qubits"
        assert output[1]["body"] == "Explain qubits."

def test_web_search_empty():
    with patch('research_agent.tools.web_search.DDGS') as MockDDGS:
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        MockDDGS.return_value = mock_instance

        output = web_search("quantum", top_k=2)
        assert output == []

def test_web_search_error():
    with patch('research_agent.tools.web_search.DDGS') as MockDDGS:
        mock_instance = MagicMock()
        mock_instance.text.side_effect = Exception("API limit exceeded")
        MockDDGS.return_value = mock_instance

        output = web_search("quantum", top_k=2)
        assert len(output) == 1
        assert output[0]["title"] == "Error"
        assert "API limit exceeded" in output[0]["body"]

def test_tool_definition():
    assert WEB_SEARCH_TOOL.name == "web_search"
    assert "query" in WEB_SEARCH_TOOL.parameters["required"]
