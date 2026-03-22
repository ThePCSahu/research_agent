import pytest
from unittest.mock import patch, MagicMock
from research_agent.models.embedding_model_client import get_embeddings

def test_get_embeddings_openai_format():
    texts = ["apple", "banana"]
    mock_response_data = {
        "model": "nomic-embed-text",
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"index": 1, "embedding": [0.4, 0.5, 0.6]}
        ]
    }
    
    with patch('research_agent.models.embedding_model_client.requests.post') as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        
        embeddings = get_embeddings(texts)
        
        mock_post.assert_called_once()
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

def test_get_embeddings_ollama_format():
    texts = ["apple"]
    mock_response_data = {
        "model": "nomic-embed-text",
        "embeddings": [
            [0.7, 0.8, 0.9]
        ]
    }
    
    with patch('research_agent.models.embedding_model_client.requests.post') as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        
        embeddings = get_embeddings(texts)
        
        assert len(embeddings) == 1
        assert embeddings[0] == [0.7, 0.8, 0.9]

def test_get_embeddings_empty():
    assert get_embeddings([]) == []

def test_get_embeddings_error():
    with patch('research_agent.models.embedding_model_client.requests.post') as mock_post:
        mock_post.side_effect = RuntimeError("Connection refused")
        with pytest.raises(RuntimeError):
            get_embeddings(["fail"])
