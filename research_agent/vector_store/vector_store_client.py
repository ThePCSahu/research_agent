import logging
from typing import List, Dict, Any

from .faiss_vector_store import FaissVectorStore
from .chunker import chunk_text
from research_agent.models.embedding_model_client import EmbeddingModelClient

logger = logging.getLogger(__name__)

class VectorStoreClient:
    """
    A unified facade for document retrieval operations.
    Handles text chunking, vector embeddings, and underlying vector database operations.
    Only exposes `add` and `search` functionalities to the rest of the application.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, dim: int = 768):
        """
        Initialize the retrieval facade.
        :param dim: The dimensionality of the vector embeddings (default 768 for nomic-embed-text).
        """
        if self._initialized:
            return
        self.vector_store = FaissVectorStore(dim=dim)
        self.embedding_client = EmbeddingModelClient()
        self._initialized = True
        
    def add(self, texts: List[str], metadata: List[Dict[str, Any]], AGENT_VECTOR_CHUNK_SIZE: int = 500, chunk_overlap: int = 50) -> None:
        """
        Chunk texts, embed them, and add to the vector store.
        
        :param texts: A list of full document texts.
        :param metadata: A list of metadata dictionaries corresponding to each document.
        :param AGENT_VECTOR_CHUNK_SIZE: Maximum size of a single text chunk.
        :param chunk_overlap: Number of characters to overlap between chunks.
        """
        if len(texts) != len(metadata):
            raise ValueError("Lengths of texts and metadata must match.")
            
        all_chunks = []
        all_metadata = []
        
        for text, meta in zip(texts, metadata):
            chunks = chunk_text(text, AGENT_VECTOR_CHUNK_SIZE=AGENT_VECTOR_CHUNK_SIZE, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
            # Duplicate the metadata for each chunk
            all_metadata.extend([meta] * len(chunks))
            
        if not all_chunks:
            logger.info("No chunks to add to the vector store.")
            return
            
        logger.info(f"Embedding and adding {len(all_chunks)} chunks to vector store.")
        embeddings = self.embedding_client.get_embeddings(all_chunks)
        self.vector_store.add(embeddings, all_chunks, all_metadata)
        
    def search(self, query: str, top_k: int = 5, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search for the most relevant text chunks matching the query.
        
        :param query: The search query string.
        :param top_k: The number of top results to return.
        :param threshold: Minimum similarity threshold (0.0 to 1.0).
        :return: A list of dictionaries containing 'text', 'metadata', and 'score'.
        """
        if not query.strip():
            return []
            
        logger.info(f"Searching vector store for query: '{query}' with threshold {threshold}")
        query_embeddings = self.embedding_client.get_embeddings([query])
        
        if not query_embeddings:
            return []
            
        return self.vector_store.search(query_embeddings[0], top_k, threshold=threshold)
