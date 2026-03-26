import pytest
from research_agent.vector_store.faiss_vector_store import FaissVectorStore

def test_faiss_vector_store():
    # Setup
    dim = 3
    store = FaissVectorStore(dim=dim)
    
    # Empty search
    assert store.search([0.1, 0.2, 0.3], top_k=2) == []
    
    # Add dummy vectors
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    texts = ["chunk1", "chunk2", "chunk3"]
    metadata = [
        {"url": "u1", "query": "q1", "chunk_id": "c1"},
        {"url": "u2", "query": "q2", "chunk_id": "c2"},
        {"url": "u3", "query": "q3", "chunk_id": "c3"},
    ]
    
    store.add(embeddings, texts, metadata)
    
    # Ensure ntotal updated
    assert store.index.ntotal == 3
    assert len(store.documents) == 3
    
    # Search for an exact match
    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "chunk1"
    assert results[0]["metadata"]["chunk_id"] == "c1"
    assert results[0]["distance"] == 0.0  # Exact match L2 distance is 0
    
    # Search for top 2
    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "chunk1"  # Closest
    
    # Mismatch dimension
    with pytest.raises(ValueError, match="Expected query dimension"):
        store.search([1.0, 0.0], top_k=1)

def test_faiss_add_mismatch_lengths():
    store = FaissVectorStore(dim=2)
    with pytest.raises(ValueError, match="Lengths.*match"):
        store.add([[1.0, 1.0]], ["t1", "t2"], [{"meta": 1}])

def test_faiss_add_mismatch_dims():
    store = FaissVectorStore(dim=3)
    with pytest.raises(ValueError, match="Expected embedding dimension"):
        store.add([[1.0, 1.0]], ["t1"], [{"meta": 1}])
