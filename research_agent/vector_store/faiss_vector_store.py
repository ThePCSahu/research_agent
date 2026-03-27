import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        
        # Attempt to use GPU if supported by the hardware and faiss installation
        try:
            if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        except Exception:
            pass # Fallback to CPU index
            
        # Maintain a parallel array of { "text": str, "metadata": dict }
        self.documents = []

    def add(self, embeddings: list[list[float]], texts: list[str], metadata: list[dict]):
        if not embeddings:
            return
            
        if len(embeddings) != len(texts) or len(texts) != len(metadata):
            raise ValueError("Lengths of embeddings, texts, and metadata must exactly match.")
            
        # Convert list of lists to float32 numpy array
        emb_array = np.array(embeddings, dtype=np.float32)
        
        if emb_array.shape[1] != self.dim:
            raise ValueError(f"Expected embedding dimension {self.dim}, got {emb_array.shape[1]}")
            
        # L2-normalize for cosine similarity via Inner Product
        faiss.normalize_L2(emb_array)
        self.index.add(emb_array)
        
        for text, meta in zip(texts, metadata):
            self.documents.append({
                "text": text,
                "metadata": meta
            })

    def search(self, query_embedding: list[float], top_k: int, threshold: float = 0.8) -> list[dict]:
        if self.index.ntotal == 0:
            return []
            
        query_array = np.array([query_embedding], dtype=np.float32)
        
        if query_array.shape[1] != self.dim:
            raise ValueError(f"Expected query dimension {self.dim}, got {query_array.shape[1]}")
            
        # L2-normalize query for cosine similarity
        faiss.normalize_L2(query_array)
        
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
            
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                if float(score) < threshold:
                    continue
                doc = self.documents[idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })
                
        return results
