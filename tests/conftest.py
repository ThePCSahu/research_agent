import os
import sys
import types
from pathlib import Path

import pytest
from unittest.mock import MagicMock


def _reset_singletons():
    """Reset all singleton instances to prevent state leaking between tests."""
    from research_agent.models.llm_client import LLMClient
    from research_agent.models.embedding_model_client import EmbeddingModelClient
    from research_agent.vector_store.vector_store_client import VectorStoreClient

    LLMClient._instance = None
    LLMClient._initialized = False
    EmbeddingModelClient._instance = None
    EmbeddingModelClient._initialized = False
    VectorStoreClient._instance = None
    VectorStoreClient._initialized = False


@pytest.fixture(autouse=True)
def reset_singletons():
    """Auto-reset singletons before every test."""
    _reset_singletons()
    yield
    _reset_singletons()


@pytest.fixture
def mock_env(monkeypatch):
    """Clear OS environ for clean testing."""
    monkeypatch.setattr(os, "environ", {})


@pytest.fixture
def mock_llm_client():
    from research_agent.models.llm_client import LLMClient
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = '["test query 1", "test query 2"]'
    return client
