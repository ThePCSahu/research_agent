import os
import pytest
from unittest.mock import patch, MagicMock

from research_agent.pipeline.research_orchestrator import ResearchOrchestrator


@pytest.fixture
def mock_deps():
    """Inject mock dependencies directly into ResearchOrchestrator."""
    mock_vsc = MagicMock()
    mock_qp = MagicMock()
    mock_analyzer = MagicMock()
    mock_engine = MagicMock()
    mock_synth = MagicMock()

    return mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth


def _make_agent(mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth):
    return ResearchOrchestrator(
        vector_store_client=mock_vsc,
        query_planner=mock_qp,
        document_analyzer=mock_analyzer,
        decision_engine=mock_engine,
        report_synthesizer=mock_synth,
    )


def test_run_agent_high_confidence(mock_deps):
    mock_vsc, mock_qp, mock_web_search_fn, mock_analyzer, mock_engine = mock_deps[:5]
    mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth = mock_deps

    mock_qp.generate_queries.return_value = ["query1"]

    mock_vsc.search.return_value = [{"text": "Sample text"}]

    mock_analyzer.analyze.return_value = {
        "insights": ["insight1"],
        "gaps": [],
        "confidence": 0.90,
    }

    mock_engine.decide_next_step.return_value = {"action": "continue", "queries": []}
    mock_synth.generate_report.return_value = "FINAL REPORT"

    with patch("research_agent.pipeline.research_orchestrator.web_search") as mock_ws, \
         patch("research_agent.pipeline.research_orchestrator.fetch_content") as mock_fc:
        mock_ws.return_value = [{"url": "http://example.com"}]
        mock_fc.return_value = {"url": "http://example.com", "content": "Sample text", "title": "Example"}

        agent = _make_agent(mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth)
        report = agent.run("Test Topic")

    assert report == "FINAL REPORT"
    mock_qp.generate_queries.assert_called_once_with("Test Topic")
    mock_ws.assert_called_once()
    mock_fc.assert_called_once_with("http://example.com")
    mock_vsc.add.assert_called_once()
    assert mock_vsc.search.call_count == 2
    mock_analyzer.analyze.assert_called_once()
    mock_engine.decide_next_step.assert_not_called()


def test_run_agent_max_iterations(mock_deps):
    mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth = mock_deps

    with patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "2", "AGENT_MAX_SEARCH_RESULTS": "3", "AGENT_TOP_K_RETRIEVAL_SIZE": "5"}):
        mock_qp.generate_queries.return_value = ["q1"]

        mock_vsc.search.return_value = [{"text": "Text"}]

        mock_analyzer.analyze.return_value = {"confidence": 0.5, "insights": [], "gaps": ["gap1"]}
        mock_engine.decide_next_step.return_value = {"action": "continue", "queries": ["q2"]}
        mock_synth.generate_report.return_value = "REPORT2"

        with patch("research_agent.pipeline.research_orchestrator.web_search") as mock_ws, \
             patch("research_agent.pipeline.research_orchestrator.fetch_content") as mock_fc:
            mock_ws.side_effect = [
                [{"url": "http://example.com/1"}],
                [{"url": "http://example.com/2"}],
            ]
            mock_fc.side_effect = [
                {"url": "http://example.com/1", "content": "Text 1"},
                {"url": "http://example.com/2", "content": "Text 2"},
            ]

            agent = _make_agent(mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth)
            report = agent.run("Topic2")

    assert report == "REPORT2"
    assert mock_analyzer.analyze.call_count == 2
    assert mock_engine.decide_next_step.call_count == 2
    assert mock_fc.call_count == 2


def test_run_agent_finish_early(mock_deps):
    mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth = mock_deps

    mock_qp.generate_queries.return_value = ["init_q"]
    mock_vsc.search.return_value = [{"text": "Data"}]
    mock_analyzer.analyze.return_value = {"confidence": 0.5}
    mock_engine.decide_next_step.return_value = {"action": "finish", "queries": []}
    mock_synth.generate_report.return_value = "REPORT_FINISHED"

    with patch("research_agent.pipeline.research_orchestrator.web_search") as mock_ws, \
         patch("research_agent.pipeline.research_orchestrator.fetch_content") as mock_fc:
        mock_ws.return_value = [{"url": "http://url.com"}]
        mock_fc.return_value = {"url": "http://url.com", "content": "Data"}

        agent = _make_agent(mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth)
        report = agent.run("Topic3")

    assert report == "REPORT_FINISHED"
    assert mock_analyzer.analyze.call_count == 1
    assert mock_engine.decide_next_step.call_count == 1


def test_run_agent_no_queries_from_decision(mock_deps):
    mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth = mock_deps

    mock_qp.generate_queries.return_value = ["init_q"]
    mock_vsc.search.return_value = [{"text": "Data"}]
    mock_analyzer.analyze.return_value = {"confidence": 0.5}
    mock_engine.decide_next_step.return_value = {"action": "continue", "queries": []}
    mock_synth.generate_report.return_value = "REPORT_EMPTY"

    with patch("research_agent.pipeline.research_orchestrator.web_search") as mock_ws, \
         patch("research_agent.pipeline.research_orchestrator.fetch_content") as mock_fc:
        mock_ws.return_value = [{"url": "http://url.com"}]
        mock_fc.return_value = {"content": "Data"}

        agent = _make_agent(mock_vsc, mock_qp, mock_analyzer, mock_engine, mock_synth)
        report = agent.run("Topic4")

    assert report == "REPORT_EMPTY"
    assert mock_analyzer.analyze.call_count == 1
    assert mock_engine.decide_next_step.call_count == 1
