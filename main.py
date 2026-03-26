from research_agent.tools.web_search import web_search
from research_agent.pipeline.research_orchestrator import ResearchOrchestrator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    agent = ResearchOrchestrator()
    final_report = agent.run("Effect of AI on children in 2030")
    print("\n=== FINAL REPORT ===\n")
    print(final_report)