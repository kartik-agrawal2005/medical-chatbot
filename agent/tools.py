"""
tools.py — FAISS Retrieval Tool for the ReAct Agent

This module provides the search_medical_kb tool that the agent can invoke
during its ReAct reasoning loop to fetch relevant medical knowledge.

Architecture Boundary:
    The LLM NEVER touches vectors directly. This tool acts as a clean 
    interface between the text world (LLM) and the vector world (FAISS):
    
        LLM → "search for: diabetes symptoms"    (plain text)
            → embed("diabetes symptoms")          (384-dim vector)  
            → FAISS.search(vector, top_k=3)        (ANN search)
            → return top-3 chunk texts             (plain text)
        LLM ← "Observation: [chunk1, chunk2, ...]" (plain text)
    
    The LLM only ever sees text in and text out. The embedding and vector 
    search happen entirely inside this tool — invisible to the agent.

Usage:
    In Phase 3's graph.py, this tool is called by the tool_execute node
    when the agent's reasoning step decides it needs medical facts.
"""

import os

from knowledge_base.embedder import MedicalEmbedder
from knowledge_base.index_builder import FAISSIndexBuilder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the pre-built vectorstore (created by scripts/build_index.py)
VECTORSTORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vectorstore",
)

# Number of chunks to retrieve per query
DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# Retrieval Tool
# ---------------------------------------------------------------------------

class MedicalKBTool:
    """
    Retrieval tool that searches the FAISS index for relevant medical chunks.
    
    This class wraps the entire retrieval pipeline:
        1. Embed the query string using MiniLM
        2. Search the FAISS HNSW index for nearest neighbors
        3. Format results as readable text for the LLM
    
    The tool uses lazy loading — the FAISS index and embedding model are 
    only loaded into memory when the tool is first called, not at import time.
    
    Attributes:
        vectorstore_dir:  Path to the directory containing the FAISS index.
        top_k:            Number of chunks to retrieve per search.
        _embedder:        Lazy-loaded MedicalEmbedder instance.
        _index_builder:   Lazy-loaded FAISSIndexBuilder instance.
    """
    
    def __init__(
        self,
        vectorstore_dir: str = VECTORSTORE_DIR,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.vectorstore_dir = vectorstore_dir
        self.top_k = top_k
        self._embedder = None
        self._index_builder = None
    
    def _load_resources(self) -> None:
        """
        Lazy-load the embedding model and FAISS index.
        
        Called on first search. After loading, subsequent searches 
        skip this step — the resources stay in memory.
        """
        if self._embedder is None:
            self._embedder = MedicalEmbedder()
        
        if self._index_builder is None:
            self._index_builder = FAISSIndexBuilder.load(self.vectorstore_dir)
    
    def search(self, query: str) -> list[dict]:
        """
        Search the medical knowledge base for relevant chunks.
        
        This is the raw search — returns structured results with scores.
        Used internally and for debugging.
        
        Args:
            query: The search query string (natural language).
        
        Returns:
            List of dicts, each containing:
                - text:        The chunk content
                - source_file: Original document filename
                - chunk_index: Position in the original document
                - score:       L2 distance (lower = more relevant)
        """
        self._load_resources()
        
        # Step 1: Embed the query → 384-dim vector
        query_vector = self._embedder.embed_text(query)
        
        # Step 2: Search FAISS index for nearest neighbors
        results = self._index_builder.search(query_vector, top_k=self.top_k)
        
        return results
    
    def search_formatted(self, query: str) -> str:
        """
        Search and return results formatted as a readable string for the LLM.
        
        This is what the ReAct agent sees as its "Observation" after 
        calling the tool. The format is designed to be clear and parseable 
        by the LLM without any special tokens.
        
        Args:
            query: The search query string (natural language).
        
        Returns:
            A formatted string containing the top-K retrieved chunks,
            each with its source attribution.
            
            Example output:
                [Source: common_conditions.txt, Chunk 3]
                Diabetes mellitus is a group of metabolic diseases...
                
                [Source: common_conditions.txt, Chunk 4]
                Treatment for Type 2 diabetes typically begins...
        """
        results = self.search(query)
        
        if not results:
            return "No relevant medical information found in the knowledge base."
        
        formatted_chunks = []
        for i, result in enumerate(results, 1):
            source = result.get("source_file", "unknown")
            chunk_idx = result.get("chunk_index", "?")
            score = result.get("score", 0.0)
            text = result.get("text", "")
            
            formatted_chunks.append(
                f"[Source: {source}, Chunk {chunk_idx} | Relevance: {score:.2f}]\n"
                f"{text}"
            )
        
        return "\n\n---\n\n".join(formatted_chunks)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded)
# ---------------------------------------------------------------------------

# Single instance shared across the agent — avoids reloading the model
# and index on every tool call within the same session
_tool_instance = None


def get_medical_kb_tool() -> MedicalKBTool:
    """
    Get or create the singleton MedicalKBTool instance.
    
    Uses module-level caching so the FAISS index and embedding model 
    are loaded only once per process, regardless of how many times 
    the agent calls the tool.
    
    Returns:
        The shared MedicalKBTool instance.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = MedicalKBTool()
    return _tool_instance


# ---------------------------------------------------------------------------
# CLI: Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  === Medical KB Retrieval Tool Test ===\n")
    
    tool = MedicalKBTool()
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How to treat high blood pressure?",
        "What causes migraine headaches?",
    ]
    
    for query in test_queries:
        print(f"  🔍 Query: \"{query}\"")
        print(f"  {'─' * 50}")
        result = tool.search_formatted(query)
        # Print first 200 chars of each result for brevity
        preview = result[:200] + "..." if len(result) > 200 else result
        print(f"  {preview}")
        print()
