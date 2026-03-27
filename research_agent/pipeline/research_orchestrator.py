from typing import Dict, List
from research_agent.utils.config import get_config_or_default
import logging

from research_agent.memory.state import AgentState
from research_agent.vector_store.vector_store_client import VectorStoreClient
from research_agent.core.query_planner import QueryPlanner
from research_agent.core.data_analyzer import DataAnalyzer
from research_agent.core.report_synthesizer import ReportSynthesizer
from research_agent.tools.web_search import web_search
from research_agent.tools.fetch_content import fetch_content
from research_agent.core.re_ranker import ReRanker
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    def __init__(self, vector_store_client=None, query_planner=None, data_analyzer=None, report_synthesizer=None):
        self.vector_store_client = vector_store_client or VectorStoreClient(dim=768)
        self.query_planner = query_planner or QueryPlanner()
        self.data_analyzer = data_analyzer or DataAnalyzer()
        self.synthesizer = report_synthesizer or ReportSynthesizer()
        self.re_ranker = ReRanker()


    def run(self, topic: str) -> str:
        """Main loop for the research agent."""
        logger.info(f"Starting research orchestrator loop for topic: '{topic}'")
        state = AgentState(topic=topic)
        
        # queries is now List[Dict[str, str]] -> [{"query": "...", "hyde": "..."}]
        queries = self.query_planner.generate_queries(topic)
        max_iterations = int(get_config_or_default("AGENT_MAX_ITERATIONS", "5"))
        max_search_results = int(get_config_or_default("AGENT_MAX_SEARCH_RESULTS", "3"))

        with ThreadPoolExecutor(max_workers=10) as executor:
            while state.iteration < max_iterations:
                logger.info(f"--- Iteration {state.iteration + 1}/{max_iterations} ---")
                
                # 1. WEB SEARCH & FETCH
                if queries:
                    self._parallel_search_and_fetch(executor, state, queries, max_search_results)

                # 2. RETRIEVAL & WORKING MEMORY
                self._parallel_retrieval(executor, state, topic, queries)
                
                # 3. ANALYSIS & DECISION
                analysis = self._analyze_iteration(state)
                confidence = analysis.get("confidence", 0.0)
                
                if confidence >= 0.85:
                    logger.info("High confidence reached. Breaking loop.")
                    break
                
                proposed_queries = analysis.get("queries", [])
                if not proposed_queries:
                    logger.info("No more queries proposed. Finishing.")
                    break

                queries = proposed_queries
                state.iteration += 1

        logger.info(f"Generating final report using {len(state.working_memory)} chunks.")
        return self.synthesizer.generate_report(state.working_memory, state)

    def _parallel_search_and_fetch(self, executor: ThreadPoolExecutor, state: AgentState, queries: list[dict], max_results: int):
        """Perform searches and content ingestion in parallel."""
        start_time = time.time()
        # queries is a list of {"query": "...", "hyde": "..."}
        search_futures = {executor.submit(web_search, q["query"], max_results): q["query"] for q in queries}
        
        all_urls = []
        for future in as_completed(search_futures):
            query = search_futures[future]
            try:
                results = future.result()
                state.record_query(query)
                for r in results:
                    if r.get("url"):
                        all_urls.append(r["url"])
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")

        def fetch_and_store(url):
            if state.record_url(url):
                content = fetch_content(url)
                if content and content.get("content"):
                    self.vector_store_client.add(
                        texts=[content["content"]], 
                        metadata=[{"url": content.get("url", url), "title": content.get("title", "")}]
                    )

        fetch_futures = [executor.submit(fetch_and_store, url) for url in set(all_urls)]
        for future in as_completed(fetch_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Fetch failed: {e}")
        
        logger.info(f"Step 1 (Parallel Search/Fetch) took {time.time() - start_time:.2f}s")

    def _parallel_retrieval(self, executor: ThreadPoolExecutor, state: AgentState, topic: str, queries: list[dict]):
        """Retrieve relevant chunks using pre-generated HyDE and update working memory."""
        start_time = time.time()
        top_k_new = int(get_config_or_default("AGENT_TOP_K_RETRIEVAL_SIZE", "15"))
        
        # Collect all search targets (original queries + pre-generated HyDE hypothetical answers)
        search_targets = [topic]
        for q in queries:
            search_targets.append(q["query"])
            search_targets.append(q["hyde"])
        
        # Deduplicate targets
        search_targets = list(set([t for t in search_targets if t]))
        
        retrieval_futures = [
            executor.submit(
                self.vector_store_client.search, 
                target, 
                top_k=top_k_new, 
                threshold=0.6
            ) for target in search_targets
        ]
        
        raw_retrieved = []
        for future in as_completed(retrieval_futures):
            try:
                raw_retrieved.extend(future.result())
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")

        # Deduplicate and initial filter
        unique_chunks_map = {c["text"]: c for c in raw_retrieved}
        unique_chunks = list(unique_chunks_map.values())
        
        # Merge with existing working memory
        all_candidate_chunks = list({c["text"]: c for c in (state.working_memory + unique_chunks)}.values())
        
        # Semantic Re-ranking
        logger.info(f"Re-ranking {len(all_candidate_chunks)} candidate chunks for topic: '{topic}'")
        all_candidate_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_candidates = all_candidate_chunks[:40]
        
        state.working_memory = self.re_ranker.re_rank(topic, top_candidates, top_n=25)
        
        logger.info(f"Step 2 (Parallel Retrieval + HyDE + Re-ranking) took {time.time() - start_time:.2f}s")

    def _analyze_iteration(self, state: AgentState) -> dict:
        """Call DataAnalyzer and update state."""
        logger.info(f"Analyzing {len(state.working_memory)} chunks.")
        analysis = self.data_analyzer.analyze([c["text"] for c in state.working_memory], state)
        
        state.insights = analysis.get("insights", [])
        state.contradictions = analysis.get("contradictions", [])
        state.gaps = analysis.get("gaps", [])
        state.sources_evaluation = analysis.get("sources_evaluation", [])
        
        logger.info(f"Analysis confidence: {analysis.get('confidence', 0.0):.2f}")
        return analysis
