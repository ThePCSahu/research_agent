import logging
import requests
from research_agent.utils.config import get_config_or_default
from .base import Tool

logger = logging.getLogger(__name__)

def web_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the web for a given query and return formatted results using SerpAPI.
    Returns:
    [
        {"url": "...", "title": "...", "body": "..."}
    ]
    """
    logger.info("web_search called with query: '%s', top_k: %d", query, top_k)

    api_key = get_config_or_default("SERPAPI_API_KEY", "")
    if not api_key:
        logger.error("SERPAPI_API_KEY is not set.")
        return [{"url": "", "title": "Error", "body": "SERPAPI_API_KEY is missing."}]

    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query
        }

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=payload, timeout=15.0)
        response.raise_for_status()
        data = response.json()

        organic_results = data.get("organic_results", [])
        
        if not organic_results:
            logger.info("No results found for query: '%s'", query)
            return []

        formatted_results = []
        seen_urls = set()
        
        for r in organic_results:
            href = r.get("link", "")
            if not href or href in seen_urls:
                continue
                
            seen_urls.add(href)
            
            title = r.get("title", "")
            body = r.get("snippet", "")
            
            formatted_results.append({
                "url": href,
                "title": title,
                "body": body
            })
            
            if len(seen_urls) >= top_k:
                break
            
        logger.info("Successfully fetched %d results", len(formatted_results))
        return formatted_results

    except Exception as e:
        logger.error("Failed to execute web search for query '%s': %s", query, e)
        return [{"url": "", "title": "Error", "body": f"Web search failed: {e}"}]


WEB_SEARCH_TOOL = Tool(
    name="web_search",
    description=(
        "Search the web for information about a given topic or query. "
        "Returns a list of search results including titles, URLs, and body."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
            },
        },
        "required": ["query"],
    },
    func=web_search,
    timeout_seconds=15.0,
)


if __name__ == "__main__":
    print(web_search("What is artificial intelligence?"))