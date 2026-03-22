from research_agent.core.query_generator import QueryGenerator
import json

_REQUIRED_FACET_TERMS = (
    "overview",
    "statistic",
    "trend",
    "geograph",
    "edge",
)


def test_query_generator_generate_queries(mock_llm_client):
    generator = QueryGenerator(llm_client=mock_llm_client)
    queries = generator.generate_queries("test topic")

    assert len(queries) == 2
    assert queries[0] == "test query 1"
    assert queries[1] == "test query 2"
    mock_llm_client.chat.assert_called_once()


def test_parse_queries_json():
    raw = '["query 1", "query 2"]'
    queries = QueryGenerator._parse_queries(raw)
    assert queries == ["query 1", "query 2"]


def test_parse_queries_markdown():
    raw = 'Here are your queries:\n```json\n["query 1", "query 2"]\n```'
    queries = QueryGenerator._parse_queries(raw)
    assert queries == ["query 1", "query 2"]


def test_parse_queries_fallback():
    raw = 'Here are some queries:\n1. query 1\n- query 2\n* "query 3"'
    queries = QueryGenerator._parse_queries(raw)
    assert "query 1" in queries
    assert "query 2" in queries
    assert "query 3" in queries


def test_dedupe_queries_removes_repetition_preserves_order():
    qs = [
        "foo bar",
        "  FOO  bar  ",
        "baz",
        "baz",
    ]
    assert QueryGenerator._dedupe_queries(qs) == ["foo bar", "baz"]


def test_generate_queries_from_gaps_invokes_llm_with_facet_rich_system_prompt(mock_llm_client):
    generator = QueryGenerator(llm_client=mock_llm_client)
    out = generator.generate_queries_from_gaps("renewable energy")

    mock_llm_client.chat.assert_called_once()
    call_kw = mock_llm_client.chat.call_args
    messages = call_kw.kwargs.get("messages") or call_kw[1].get("messages")
    assert messages is not None
    system = messages[0]["content"]
    for term in _REQUIRED_FACET_TERMS:
        assert term.lower() in system.lower()

    assert out == ["test query 1", "test query 2"]


def test_generate_queries_from_gaps_returns_diverse_non_repetitive_list(mock_llm_client):
    """Distinct angles: no duplicate strings after normalization."""
    diverse = [
        "topic overview and definition",
        "topic key statistics 2024",
        "topic market trends forecast",
        "topic adoption by region Asia Europe",
        "topic limitations failures edge cases",
    ]
    mock_llm_client.chat.return_value = json.dumps(diverse)

    generator = QueryGenerator(llm_client=mock_llm_client)
    queries = generator.generate_queries_from_gaps("some topic")

    assert len(queries) == len(diverse)
    assert len(set(q.lower() for q in queries)) == len(queries)


def test_generate_queries_from_gaps_dedupes_when_llm_repeats(mock_llm_client):
    raw = '["same angle", "SAME  angle", "unique query"]'
    mock_llm_client.chat.return_value = raw

    generator = QueryGenerator(llm_client=mock_llm_client)
    queries = generator.generate_queries_from_gaps("x")

    assert queries == ["same angle", "unique query"]
