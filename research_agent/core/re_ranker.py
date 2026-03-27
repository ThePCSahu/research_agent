import logging
from typing import List, Dict
from research_agent.models.llm_client import LLMClient

logger = logging.getLogger(__name__)

_RE_RANK_PROMPT = """\
You are an expert information retrieval specialist. 
Your task is to rank the following chunks based on their relevance and informational value to the research topic: "{topic}"

For each chunk, assign a relevance score between 0.0 and 1.0.
A score of 1.0 means the chunk is extremely relevant and contains crucial facts.
A score of 0.0 means the chunk is completely irrelevant.

Format your response as a JSON object where keys are the chunk indices (0, 1, 2...) and values are the scores.
Example: {{"0": 0.95, "1": 0.4, "2": 0.8}}

Chunks:
{chunks_text}

JSON Response:
"""

class ReRanker:
    def __init__(self, llm_client=None):
        self.llm = llm_client or LLMClient()

    def re_rank(self, topic: str, chunks: List[Dict], top_n: int = 15) -> List[Dict]:
        """Recognize and re-rank chunks using LLM scoring."""
        if not chunks:
            return []
        
        logger.info(f"Re-ranking {len(chunks)} chunks for topic: '{topic}'")
        
        # Prepare chunk text for LLM
        chunks_text = ""
        for i, c in enumerate(chunks):
            chunks_text += f"\n--- Chunk {i} ---\n{c['text']}\n"
        
        messages = [
            {"role": "system", "content": _RE_RANK_PROMPT.format(topic=topic, chunks_text=chunks_text)},
        ]
        
        try:
            raw = self.llm.chat(messages=messages, temperature=0.0)
            # Simple extractor for json in case LLM adds markdown
            import json
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                logger.warning("Failed to parse re-ranking scores. Returning original order.")
                return chunks[:top_n]
            
            scores = json.loads(match.group())
            
            # Update scores in chunks
            for i, c in enumerate(chunks):
                score_key = str(i)
                if score_key in scores:
                    # Blend vector score with LLM score (optional, here we just use LLM score)
                    c["re_rank_score"] = float(scores[score_key])
                else:
                    c["re_rank_score"] = 0.0
            
            # Sort by re_rank_score
            ranked = sorted(chunks, key=lambda x: x.get("re_rank_score", 0), reverse=True)
            return ranked[:top_n]
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return chunks[:top_n]
