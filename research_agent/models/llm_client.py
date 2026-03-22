from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from research_agent.utils.config import get_config, get_config_or_default


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MODEL = "llama3.1:8b"


class LLMClient:
    """
    Thin REST client for chat-completion-style LLMs.

    Responsibilities:
    - Build HTTP payloads
    - Handle authentication and timeouts
    - Return raw assistant message content as a string

    All required configuration is resolved via the config helper:
      .env file  →  OS environment variable  →  RuntimeError
      
    Implemented as a Singleton to prevent duplicate instantiations.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        if self._initialized:
            return
            
        # Required – will raise ConfigError if missing from both .env and env
        self.api_key = api_key or get_config_or_default("LLM_API_KEY", "")

        # Optional – fall back to sensible defaults
        self.base_url = base_url or get_config_or_default(
            "LLM_BASE_URL", _DEFAULT_BASE_URL
        )
        self.model = model or get_config_or_default(
            "LLM_MODEL", _DEFAULT_MODEL
        )
        self.request_timeout = request_timeout or float(
            get_config_or_default("LLM_REQUEST_TIMEOUT", str(_DEFAULT_TIMEOUT))
        )
        self._initialized = True

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> str:
        """
        Send a chat completion request and return the assistant's message content.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "LLM Request - Model: %s, Temperature: %s, Message count: %d",
            self.model, temperature, len(messages),
        )
        logger.debug("LLM Request payload: %s", json.dumps(payload, indent=2))

        try:
            logger.info("Sending HTTP POST to %s", self.base_url)
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            )
            logger.info("LLM API response status: %s", response.status_code)
        except requests.RequestException as exc:
            logger.error("HTTP request to LLM failed: %s", exc)
            raise RuntimeError(f"HTTP request to LLM failed: {exc}") from exc

        if response.status_code != 200:
            logger.error(
                "LLM API returned status %s: %s",
                response.status_code, response.text,
            )
            raise RuntimeError(
                f"LLM API returned status {response.status_code}: {response.text}"
            )

        data = response.json()
        logger.debug("LLM API raw response: %s", json.dumps(data, indent=2))

        try:
            content = data["choices"][0]["message"]["content"] or ""
            logger.info("LLM Response - Content length: %d characters", len(content))
            logger.debug(
                "LLM Response content: %s",
                content[:500] + ("..." if len(content) > 500 else ""),
            )
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "Unexpected response structure from LLM API: %s", json.dumps(data),
            )
            raise RuntimeError(
                f"Unexpected response structure from LLM API: {json.dumps(data)}"
            ) from exc

        if not isinstance(content, str):
            logger.error("Expected content to be a string, got %s", type(content))
            raise RuntimeError(
                f"Expected content to be a string, got {type(content)}"
            )

        return content
