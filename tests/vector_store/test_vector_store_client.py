import pytest
from unittest.mock import patch, MagicMock
from research_agent.vector_store.vector_store_client import VectorStoreClient


def test_vector_store_client_add_and_search():
    with patch('research_agent.vector_store.vector_store_client.EmbeddingModelClient') as MockEmbed:
        mock_embed_instance = MagicMock()
        MockEmbed.return_value = mock_embed_instance

        # When we add, return embeddings for the chunks produced
        mock_embed_instance.get_embeddings.return_value = [[1.0, 0.0, 0.0]]

        client = VectorStoreClient(dim=3)
        client.add(["A B C"], [{"doc": 1}])

        # verify it was added
        assert client.vector_store.index.ntotal == 1

        # Now search
        mock_embed_instance.get_embeddings.return_value = [[1.0, 0.0, 0.0]]
        results = client.search("A B C", top_k=1)

        assert len(results) == 1
        assert "text" in results[0]
        assert results[0]["metadata"] == {"doc": 1}


def test_vector_store_client_add_mismatch():
    with patch('research_agent.vector_store.vector_store_client.EmbeddingModelClient'):
        client = VectorStoreClient(dim=3)
        with pytest.raises(ValueError, match="Lengths.*match"):
            client.add(["A"], [{"doc": 1}, {"doc": 2}])


def test_vector_store_client_empty_search():
    with patch('research_agent.vector_store.vector_store_client.EmbeddingModelClient') as MockEmbed:
        mock_embed_instance = MagicMock()
        MockEmbed.return_value = mock_embed_instance

        client = VectorStoreClient(dim=3)
        assert client.search("") == []
