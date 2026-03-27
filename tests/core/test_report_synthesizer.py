"""Tests for research_agent.core.report_synthesizer."""

from __future__ import annotations

from unittest.mock import patch

from research_agent.core.report_synthesizer import (
    ReportSynthesizer,
    _format_chunk_bundle,
    _normalize_chunk,
    _strip_wrapping_fences,
)
from research_agent.memory.state import AgentState


def test_normalize_chunk_merges_metadata_and_top_level():
    text, meta = _normalize_chunk(
        {
            "content": "  body  ",
            "metadata": {"title": "A"},
            "url": "https://example.com",
        }
    )
    assert text == "body"
    assert meta["title"] == "A"
    assert meta["url"] == "https://example.com"


def test_format_chunk_bundle_includes_reference_numbers_and_excerpts():
    chunks = [
        {
            "text": "First fact.",
            "metadata": {"title": "Doc A", "url": "https://a.example"},
        },
        {
            "content": "Second fact.",
            "title": "Doc B",
            "url": "https://b.example",
        },
    ]
    bundle = _format_chunk_bundle(chunks)
    assert "[1]" in bundle and "Doc A" in bundle and "https://a.example" in bundle
    assert "[2]" in bundle and "Doc B" in bundle
    assert "First fact." in bundle and "Second fact." in bundle


def test_format_chunk_bundle_skips_empty_text():
    assert _format_chunk_bundle([{"text": "", "url": "https://x"}]) == ""


def test_strip_wrapping_fences():
    raw = "```markdown\n# Hi\n\nBody.\n```"
    assert _strip_wrapping_fences(raw).startswith("# Hi")


def test_generate_report_empty_chunks_no_llm():
    synth = ReportSynthesizer()
    state = AgentState(topic="test")
    with patch.object(synth.llm, "chat") as mock_chat:
        out = synth.generate_report([], state)
        mock_chat.assert_not_called()
    assert "No source excerpts" in out


def test_generate_report_calls_llm(mock_llm_client):
    mock_llm_client.chat.return_value = "# Report\n\nHello [1].\n\n## Sources\n\n[1] **T** — https://z"
    chunks = [
        {"text": "Evidence.", "metadata": {"title": "T", "url": "https://z"}},
    ]
    synth = ReportSynthesizer(llm_client=mock_llm_client)
    state = AgentState(topic="T")
    out = synth.generate_report(chunks, state)

    mock_llm_client.chat.assert_called_once()
    assert "# Report" in out
    assert "[1]" in out
