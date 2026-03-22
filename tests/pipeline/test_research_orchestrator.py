import os
import pytest
from unittest.mock import patch, MagicMock

from research_agent.pipeline.research_orchestrator import ResearchOrchestrator

@pytest.fixture
def mock_all():
    with patch("research_agent.pipeline.research_orchestrator.VectorStoreClient") as mock_vsc_class, \
         patch("research_agent.pipeline.research_orchestrator.QueryGenerator") as mock_qg_class, \
         patch("research_agent.pipeline.research_orchestrator.web_search") as mock_web_search, \
         patch("research_agent.pipeline.research_orchestrator.fetch_content") as mock_fetch, \
         patch("research_agent.pipeline.research_orchestrator.DocumentAnalyzer") as mock_analyzer_class, \
         patch("research_agent.pipeline.research_orchestrator.DecisionEngine") as mock_engine_class, \
         patch("research_agent.pipeline.research_orchestrator.ReportSynthesizer") as mock_synth_class:
        
        mock_vsc = MagicMock()
        mock_vsc_class.return_value = mock_vsc
        
        mock_qg = MagicMock()
        mock_qg_class.return_value = mock_qg
        
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        mock_synth = MagicMock()
        mock_synth_class.return_value = mock_synth

        yield (
            mock_vsc,
            mock_qg,
            mock_web_search,
            mock_fetch,
            mock_analyzer,
            mock_engine,
            mock_synth
        )

def test_run_agent_high_confidence(mock_all):
    mock_vsc, mock_qg, mock_web_search, mock_fetch, mock_analyzer, mock_engine, mock_synth = mock_all

    mock_qg.generate_queries.return_value = ["query1"]
    mock_web_search.return_value = [{"url": "http://example.com"}]
    mock_fetch.return_value = {"url": "http://example.com", "content": "Sample text", "title": "Example"}
    
    mock_vsc.search.return_value = [{"text": "Sample text"}]
    
    mock_analyzer.analyze.return_value = {
        "insights": ["insight1"],
        "gaps": [],
        "confidence": 0.90 # high confidence break
    }
    
    # Should not reach here due to high confidence break
    mock_engine.decide_next_step.return_value = {"action": "continue", "queries": []}
    
    mock_synth.generate_report.return_value = "FINAL REPORT"

    agent = ResearchOrchestrator()
    report = agent.run("Test Topic")

    assert report == "FINAL REPORT"
    mock_qg.generate_queries.assert_called_once_with("Test Topic")
    mock_web_search.assert_called_once()
    mock_fetch.assert_called_once_with("http://example.com")
    mock_vsc.add.assert_called_once()
    # 1 search after fetch, 1 search at the end
    assert mock_vsc.search.call_count == 2
    mock_analyzer.analyze.assert_called_once()
    mock_engine.decide_next_step.assert_not_called()

def test_run_agent_max_iterations(mock_all):
    mock_vsc, mock_qg, mock_web_search, mock_fetch, mock_analyzer, mock_engine, mock_synth = mock_all

    # Mock low confidence across all iterations
    with patch.dict(os.environ, {"MAX_ITERATIONS": "2", "MAX_SEARCH_RESULTS": "3", "AGENT_TOP_K_RETRIEVAL_SIZE": "5"}):
        mock_qg.generate_queries.return_value = ["q1"]
        
        mock_web_search.side_effect = [
            [{"url": "http://example.com/1"}], # Iteration 1
            [{"url": "http://example.com/2"}]  # Iteration 2
        ]
        
        mock_fetch.side_effect = [
            {"url": "http://example.com/1", "content": "Text 1"},
            {"url": "http://example.com/2", "content": "Text 2"}
        ]
        
        mock_vsc.search.return_value = [{"text": "Text"}]
        
        mock_analyzer.analyze.return_value = {"confidence": 0.5, "insights": [], "gaps": ["gap1"]}
        mock_engine.decide_next_step.return_value = {"action": "continue", "queries": ["q2"]}
        
        mock_synth.generate_report.return_value = "REPORT2"

        agent = ResearchOrchestrator()
        report = agent.run("Topic2")

        assert report == "REPORT2"
        # Two iterations means analyze called twice
        assert mock_analyzer.analyze.call_count == 2
        assert mock_engine.decide_next_step.call_count == 2
        
        # Finally it should perform 1 extra call to fetch for iteration 2
        assert mock_fetch.call_count == 2

def test_run_agent_finish_early(mock_all):
    mock_vsc, mock_qg, mock_web_search, mock_fetch, mock_analyzer, mock_engine, mock_synth = mock_all

    mock_qg.generate_queries.return_value = ["init_q"]
    mock_web_search.return_value = [{"url": "http://url.com"}]
    mock_fetch.return_value = {"url": "http://url.com", "content": "Data"}
    
    mock_vsc.search.return_value = [{"text": "Data"}]
    
    mock_analyzer.analyze.return_value = {"confidence": 0.5}
    # Decision says finish
    mock_engine.decide_next_step.return_value = {"action": "finish", "queries": []}
    mock_synth.generate_report.return_value = "REPORT_FINISHED"

    agent = ResearchOrchestrator()
    report = agent.run("Topic3")

    assert report == "REPORT_FINISHED"
    assert mock_analyzer.analyze.call_count == 1
    assert mock_engine.decide_next_step.call_count == 1

def test_run_agent_no_queries_from_decision(mock_all):
    mock_vsc, mock_qg, mock_web_search, mock_fetch, mock_analyzer, mock_engine, mock_synth = mock_all

    mock_qg.generate_queries.return_value = ["init_q"]
    mock_web_search.return_value = [{"url": "http://url.com"}]
    mock_fetch.return_value = {"content": "Data"}
    
    mock_vsc.search.return_value = [{"text": "Data"}]
    
    mock_analyzer.analyze.return_value = {"confidence": 0.5}
    # Decision returns empty queries, but not strictly 'finish' action
    mock_engine.decide_next_step.return_value = {"action": "continue", "queries": []}
    mock_synth.generate_report.return_value = "REPORT_EMPTY"

    agent = ResearchOrchestrator()
    report = agent.run("Topic4")

    assert report == "REPORT_EMPTY"
    assert mock_analyzer.analyze.call_count == 1
    assert mock_engine.decide_next_step.call_count == 1
