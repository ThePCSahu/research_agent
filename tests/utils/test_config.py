import pytest
from research_agent.utils.config import get_config, get_config_or_default, ConfigError

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


def test_resolve_env_value_simple(mock_env, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "${OTHER_KEY}")
    monkeypatch.setenv("OTHER_KEY", "resolved_value")
    assert get_config("TEST_KEY") == "resolved_value"


def test_resolve_env_value_recursive(mock_env, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "${LEVEL1}")
    monkeypatch.setenv("LEVEL1", "${LEVEL2}")
    monkeypatch.setenv("LEVEL2", "final_value")
    assert get_config("TEST_KEY") == "final_value"


def test_resolve_env_value_missing_reference_raises_error(mock_env, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "${MISSING_REF}")
    with pytest.raises(ConfigError, match="Referenced environment variable 'MISSING_REF' not found"):
        get_config("TEST_KEY")


def test_resolve_env_value_no_placeholder(mock_env, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "plain_value")
    assert get_config("TEST_KEY") == "plain_value"


def test_get_config_or_default_with_placeholder(mock_env, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "${OTHER_KEY}")
    monkeypatch.setenv("OTHER_KEY", "resolved")
    assert get_config_or_default("TEST_KEY", "default") == "resolved"


def test_get_config_or_default_fallback_with_placeholder(mock_env, monkeypatch):
    # If key not set, default has placeholder, should resolve default
    monkeypatch.setenv("REF", "resolved_default")
    assert get_config_or_default("MISSING", "${REF}") == "resolved_default"
