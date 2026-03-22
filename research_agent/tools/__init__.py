from typing import Dict
from .base import Tool, ToolExecutionError
from .web_search import WEB_SEARCH_TOOL
from .fetch_content import FETCH_CONTENT_TOOL

def get_tools() -> Dict[str, Tool]:
    """
    Build a default registry of core research tools.
    """
    return {
        "web_search": WEB_SEARCH_TOOL,
        "fetch_content": FETCH_CONTENT_TOOL,
    }

__all__ = ["Tool", "ToolExecutionError", "get_tools"]
