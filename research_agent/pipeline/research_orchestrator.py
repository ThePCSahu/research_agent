from research_agent.utils.config import get_config_or_default
import logging

from research_agent.memory.state import AgentState
from research_agent.vector_store.vector_store_client import VectorStoreClient
from research_agent.core.query_planner import QueryPlanner
from research_agent.core.document_analyzer import DocumentAnalyzer
from research_agent.core.decision_engine import DecisionEngine
from research_agent.core.report_synthesizer import ReportSynthesizer
from research_agent.tools.web_search import web_search
from research_agent.tools.fetch_content import fetch_content

logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    def __init__(self, vector_store_client=None, query_planner=None, document_analyzer=None, decision_engine=None, report_synthesizer=None):
        self.vector_store_client = vector_store_client or VectorStoreClient(dim=768)
        self.query_planner = query_planner or QueryPlanner()
        self.analyzer = document_analyzer or DocumentAnalyzer()
        self.decision_engine = decision_engine or DecisionEngine(query_planner=self.query_planner)
        self.synthesizer = report_synthesizer or ReportSynthesizer()

    def run(self, topic: str) -> str:
        """
        Main loop for the research agent.
        """
        logger.info(f"Starting research orchestrator loop for topic: '{topic}'")
        state = AgentState()
        
        queries = self.query_planner.generate_queries(topic)
        max_iterations = int(get_config_or_default("AGENT_MAX_ITERATIONS", "5"))
        max_search_results = int(get_config_or_default("AGENT_MAX_SEARCH_RESULTS", "3"))

        while state.iteration < max_iterations:
            logger.info(f"--- Iteration {state.iteration + 1}/{max_iterations} ---")
            
            for query in queries:
                logger.info(f"Processing query: '{query}'")
                state.record_query(query)
                
                # SEARCH
                results = web_search(query, max_search_results)
                
                for r in results:
                    url = r.get("url")
                    if not url:
                        continue
                    
                    # record_url returns False if the URL was already in the set
                    if not state.record_url(url):
                        continue
                        
                    # FETCH
                    content = fetch_content(url)
                    
                    # STORE
                    if content and "content" in content and content["content"]:
                        # Create a single document representation
                        text_data = content["content"]
                        metadata_dict = {
                            "url": content.get("url", url),
                            "title": content.get("title", ""),
                        }
                        self.vector_store_client.add(texts=[text_data], metadata=[metadata_dict])
                    
            # RETRIEVE (using the provided topic as the query context)
            top_k = int(get_config_or_default("AGENT_TOP_K_RETRIEVAL_SIZE", "5"))
            retrieved_chunks = self.vector_store_client.search(topic, top_k=top_k)
            
            # ANALYZE
            if retrieved_chunks:
                chunk_texts = [c["text"] for c in retrieved_chunks]
                analysis = self.analyzer.analyze(chunk_texts)
                
                state.insights = analysis.get("insights", [])
                state.gaps = analysis.get("gaps", [])
                
                confidence = analysis.get("confidence", 0.0)
                logger.info(f"Analysis confidence: {confidence:.2f}")
                
                if confidence > 0.85:
                    logger.info("High confidence reached. Breaking loop.")
                    break
            else:
                logger.warning("No chunks retrieved. Skipping analysis.")

            # DECIDE NEXT QUERIES
            decision = self.decision_engine.decide_next_step(state)
            queries = decision.get("queries", [])

            if decision.get("action") == "finish":
                logger.info("Decision engine signaled to finish. Breaking loop.")
                break

            if not queries:
                logger.info("No new queries generated. Breaking loop.")
                break

            state.iteration += 1

        # FINAL RETRIEVAL
        logger.info("Performing final retrieval.")
        final_chunks = self.vector_store_client.search(topic, top_k=15)

        # REPORT
        logger.info("Generating final report.")
        report = self.synthesizer.generate_report(final_chunks)

        return report

