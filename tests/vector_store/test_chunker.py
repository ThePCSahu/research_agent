import pytest
from research_agent.retrieval.chunker import chunk_text

def test_chunk_text_basic():
    text = "A" * 1000
    chunks = chunk_text(text, AGENT_VECTOR_CHUNK_SIZE=500, chunk_overlap=50)
    
    # 1000 chars divided by 500 length with 50 overlap:
    # Chunk 1: 0-500. Next chunk starts at 450.
    # Chunk 2: 450-950. Next chunk starts at 900.
    # Chunk 3: 900-1000.
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500
    assert len(chunks[2]) == 100
    
    # Verify exact overlap logic on a single character space
    assert chunks[1][:50] == chunks[0][-50:]

def test_chunk_text_paragraphs():
    # 3 paragraphs of exactly 100 A's
    text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
    chunks = chunk_text(text, AGENT_VECTOR_CHUNK_SIZE=150, chunk_overlap=20)
    
    assert len(chunks) == 3
    assert chunks[0] == "A" * 100
    assert chunks[1] == "B" * 100
    assert chunks[2] == "C" * 100

def test_chunk_text_words():
    # Long paragraph of words
    words = ["word"] * 50
    text = " ".join(words) # 249 chars long
    chunks = chunk_text(text, AGENT_VECTOR_CHUNK_SIZE=100, chunk_overlap=20)
    
    for c in chunks:
        assert len(c) <= 100

def test_chunk_text_empty():
    assert chunk_text("") == []

def test_chunk_text_small():
    text = "Short text"
    assert chunk_text(text, AGENT_VECTOR_CHUNK_SIZE=500, chunk_overlap=50) == [text]
