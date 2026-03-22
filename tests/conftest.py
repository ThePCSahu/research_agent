import os
import sys
import types
from pathlib import Path

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_env(monkeypatch):
    """Clear OS environ for clean testing."""
    monkeypatch.setattr(os, "environ", {})

@pytest.fixture
def mock_llm_client():
    from research_agent.llm_client import LLMClient
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = '["test query 1", "test query 2"]'
    return client
