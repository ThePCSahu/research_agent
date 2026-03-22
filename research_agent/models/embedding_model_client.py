from typing import Optional
import json
import logging
from typing import List
import requests

from research_agent.utils.config import get_config_or_default

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL = "http://localhost:11434/v1/embeddings"
_DEFAULT_TIMEOUT = 20.0
_DEFAULT_MODEL = "nomic-embed-text"


class EmbeddingModelClient:
    """
    Singleton client for generating vector embeddings.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingModelClient, cls).__new__(cls)
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
        self.api_key = api_key or get_config_or_default("EMBEDDING_MODEL_API_KEY", "")

        # Optional – fall back to sensible defaults
        self.base_url = base_url or get_config_or_default(
            "EMBEDDING_MODEL_BASE_URL", _DEFAULT_BASE_URL
        )
        self.model = model or get_config_or_default(
            "EMBEDDING_MODEL", _DEFAULT_MODEL
        )
        self.request_timeout = request_timeout or float(
            get_config_or_default("EMBEDDING_MODEL_REQUEST_TIMEOUT", str(_DEFAULT_TIMEOUT))
        )
        self._initialized = True

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of texts using the configured 
        Embedding API (defaults to local Ollama with nomic-embed-text).
        """
        if not texts:
            return []

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": texts
        }

        logger.info("Requesting embeddings for %d texts from %s using model %s", len(texts), self.base_url, self.model)

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # OpenAI compatible /v1/embeddings structure
            if "data" in data and isinstance(data["data"], list):
                # Sort by index just in case API returns them scrambled
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                embeddings = [item["embedding"] for item in sorted_data]
                return embeddings
                
            # Ollama native /api/embed structure fallback
            elif "embeddings" in data:
                return data["embeddings"]
                
            else:
                logger.error("Unrecognized embedding API response format: %s", json.dumps(data)[:200])
                raise RuntimeError("Unrecognized embedding API response format.")

        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch embeddings: %s", e)
            raise RuntimeError(f"Embedding request failed: {e}") from e
