import pytest
from research_agent.models.llm_client import LLMClient
from research_agent.config import ConfigError

def test_llm_client_missing_api_key(mock_env):
    with pytest.raises(ConfigError):
        LLMClient()

def test_llm_client_with_api_key(mock_env, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test_key")
    client = LLMClient()
    assert client.api_key == "test_key"
    assert client.base_url == "https://api.openai.com/v1/chat/completions"
    assert client.model == "gpt-4o"
    assert client.request_timeout == 60.0
