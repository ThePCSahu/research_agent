import pytest
from research_agent.config import get_config, get_config_or_default, ConfigError

def test_get_config_success(mock_env, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test_value")
    assert get_config("TEST_KEY") == "test_value"

def test_get_config_missing_raises_error(mock_env):
    with pytest.raises(ConfigError):
        get_config("MISSING_KEY")

def test_get_config_or_default_success(mock_env, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test_value")
    assert get_config_or_default("TEST_KEY", "default") == "test_value"

def test_get_config_or_default_fallback(mock_env):
    assert get_config_or_default("MISSING_KEY", "default") == "default"
