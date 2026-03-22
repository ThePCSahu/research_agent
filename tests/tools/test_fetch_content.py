import pytest
from unittest.mock import patch, MagicMock
from research_agent.tools.fetch_content import fetch_content, FETCH_CONTENT_TOOL
import requests

def test_fetch_content_success():
    html_content = b"""
    <html>
        <head><title>Test Page</title><style>body {color: red;}</style></head>
        <body>
            <nav>Menu Items</nav>
            <h1>Main Title</h1>
            <p>This is a paragraph with <a href="#">a link</a>.</p>
            <script>alert('Hello');</script>
        </body>
    </html>
    """

    with patch('research_agent.tools.fetch_content.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.content = html_content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        output = fetch_content("https://example.com")
        
        # Verify it hit requests.get
        mock_get.assert_called_once()
        
        assert output["url"] == "https://example.com"
        assert output["title"] == "Test Page"
        
        # Ensure scripts and styles are filtered out but semantic content remains
        assert "Main Title" in output["content"]
        assert "This is a paragraph with a link" in output["content"]
        
        # Ensure excluded content is not in output
        assert "body {color: red;}" not in output["content"]
        assert "alert('Hello');" not in output["content"]
        assert "Menu Items" not in output["content"]

def test_fetch_content_http_error():
    with patch('research_agent.tools.fetch_content.requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        output = fetch_content("https://example.com/notfound")
        assert output["url"] == "https://example.com/notfound"
        assert output["title"] == "Error"
        assert "Failed to fetch content" in output["content"]
        assert "404 Not Found" in output["content"]

def test_tool_definition():
    assert FETCH_CONTENT_TOOL.name == "fetch_content"
    assert "url" in FETCH_CONTENT_TOOL.parameters["required"]
