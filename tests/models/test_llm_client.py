import pytest
from unittest.mock import patch
from research_agent.models.llm_client import LLMClient
from research_agent.utils.config import ConfigError


def test_llm_client_with_explicit_api_key():
    client = LLMClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.base_url == "https://api.openai.com/v1/chat/completions"
    assert client.model == "llama3.1:8b"
    assert client.request_timeout == 60.0


def test_llm_client_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env_key")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    client = LLMClient()
    assert client.api_key == "env_key"
    assert client.model == "custom-model"
