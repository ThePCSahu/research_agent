import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.vector_store_client import VectorStoreClient

def test_retrieval_facade_add_and_search():
    facade = VectorStoreClient(dim=3)
    
    # Mock get_embeddings to just return deterministic embeddings for testing
    with patch('research_agent.retrieval.facade.get_embeddings') as mock_embed:
        # 1 document -> 2 chunks if chunk size is small
        # Actually, let's just make the mock return correct sized lists
        # _chunk_text will split "A B C" into something.
        # Let's mock _chunk_text as well to make it predictable, or just depend on its exact logic
        
        # When we add, mock_embed should return embeddings for whatever chunks are produced.
        # "A B C" -> Let's say it makes 1 chunk if AGENT_VECTOR_CHUNK_SIZE=500
        mock_embed.return_value = [[1.0, 0.0, 0.0]]
        
        facade.add(["A B C"], [{"doc": 1}])
        
        # verify it was added
        assert facade.vector_store.index.ntotal == 1
        
        # Now search
        mock_embed.return_value = [[1.0, 0.0, 0.0]]
        results = facade.search("A B C", top_k=1)
        
        assert len(results) == 1
        assert "text" in results[0]
        assert results[0]["metadata"] == {"doc": 1}

def test_retrieval_facade_add_mismatch():
    facade = RetrievalFacade(dim=3)
    with pytest.raises(ValueError, match="Lengths.*match"):
        facade.add(["A"], [{"doc": 1}, {"doc": 2}])

def test_retrieval_facade_empty_search():
    facade = RetrievalFacade(dim=3)
    with patch('research_agent.retrieval.facade.get_embeddings', return_value=[]):
        assert facade.search("") == []
