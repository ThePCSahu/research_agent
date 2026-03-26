import sys
import logging
from unittest.mock import patch, MagicMock

from research_agent.pipeline.research_orchestrator import ResearchOrchestrator

logging.basicConfig(level=logging.INFO)

def main():
    with patch("research_agent.pipeline.research_orchestrator.QueryPlanner") as mock_qg_class, \
         patch("research_agent.pipeline.research_orchestrator.web_search") as mock_web_search, \
         patch("research_agent.pipeline.research_orchestrator.fetch_content") as mock_fetch, \
         patch("research_agent.pipeline.research_orchestrator.VectorStoreClient") as mock_vsc_class, \
         patch("research_agent.pipeline.research_orchestrator.DocumentAnalyzer") as mock_analyzer_class, \
         patch("research_agent.pipeline.research_orchestrator.DecisionEngine") as mock_decision_class, \
         patch("research_agent.pipeline.research_orchestrator.ReportSynthesizer") as mock_report_class:
        
        mock_qg_class.return_value.generate_queries.return_value = ["dummy query"]
        mock_web_search.return_value = [{"url": "http://dummy.url"}]
        mock_fetch.return_value = {"url": "http://dummy.url", "content": "Dummy Content", "title": "Dummy"}
        
        mock_vsc = mock_vsc_class.return_value
        mock_vsc.search.return_value = [{"text": "Dummy Content", "metadata": {"url": "http://dummy.url"}}]
        
        mock_analyze_mock = mock_analyzer_class.return_value
        mock_analyze_mock.analyze.return_value = {
            "insights": ["Dummy insight"],
            "gaps": ["Dummy gap"],
            "confidence": 0.9
        }
        
        mock_decide_mock = mock_decision_class.return_value
        mock_decide_mock.decide_next_step.return_value = {"action": "finish", "queries": []}
        
        mock_report_mock = mock_report_class.return_value
        mock_report_mock.generate_report.return_value = "FINAL MOCK REPORT"
        
        print("Starting mock E2E Run...")
        agent = ResearchOrchestrator()
        report = agent.run("Dummy Topic")
        print("Report received:")
        print(report)

if __name__ == "__main__":
    main()
