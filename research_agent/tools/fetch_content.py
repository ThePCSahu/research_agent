import logging
import requests
from bs4 import BeautifulSoup
from .base import Tool

logger = logging.getLogger(__name__)

def fetch_content(url: str) -> dict:
    """
    Fetch clean readable text from webpage.
    Removes HTML tags, scripts, and styles.
    Returns:
        {"url": str, "title": str, "content": str}
    """
    logger.info("fetch_content called with url: '%s'", url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract title before stripping tags
        title = soup.title.string if soup.title else ""
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            script_or_style.extract()
            
        # Extract text, replace multiple spaces with single space
        text = soup.get_text(separator=' ', strip=True)
        
        logger.info("Successfully fetched %d characters from '%s'", len(text), url)
        return {
            "url": url,
            "title": title.strip() if title else "",
            "content": text
        }

    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch content from URL '%s': %s", url, e)
        return {"url": url, "title": "Error", "content": f"Failed to fetch content: HTTP request error - {e}"}
    except Exception as e:
        logger.error("Failed to parse content from URL '%s': %s", url, e)
        return {"url": url, "title": "Error", "content": f"Failed to extract content: {e}"}


FETCH_CONTENT_TOOL = Tool(
    name="fetch_content",
    description=(
        "Fetch clean readable text from a given webpage URL. "
        "Useful for reading articles, documentation, or any text-heavy webpage."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to fetch content from.",
            }
        },
        "required": ["url"],
    },
    func=fetch_content,
    timeout_seconds=20.0,
)
