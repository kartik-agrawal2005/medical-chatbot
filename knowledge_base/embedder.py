"""
embedder.py — MiniLM Embedding Wrapper

Wraps the sentence-transformers/all-MiniLM-L6-v2 model to convert plain text
chunks into 384-dimensional dense vectors for FAISS indexing.

Key Concepts (mapping to your existing knowledge):
    - An embedding is essentially a hash function, but instead of mapping to
      a single bucket, it maps text to a point in 384-dimensional space where
      "similar meaning" = "nearby points" (cosine similarity / L2 distance).
    - Unlike cryptographic hashes, these embeddings PRESERVE semantic 
      relationships: embed("headache") will be close to embed("migraine")
      in vector space.
    - The model runs entirely locally on CPU — no API calls needed.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Embedding Model Wrapper
# ---------------------------------------------------------------------------

class MedicalEmbedder:
    """
    Wrapper around the MiniLM-L6-v2 sentence transformer.
    
    This class handles model loading and provides a clean interface for 
    embedding single strings or batches of text. The model outputs 
    384-dimensional float32 vectors.
    
    Attributes:
        model_name:     HuggingFace model identifier.
        dimension:      Output embedding dimension (384 for MiniLM-L6-v2).
        _model:         Lazy-loaded SentenceTransformer instance.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384  # MiniLM-L6-v2 output dimension
        self._model = None
    
    def _load_model(self) -> SentenceTransformer:
        """
        Lazy-load the model on first use.
        
        This avoids loading the ~80MB model into memory until we actually
        need it, which keeps import time fast.
        """
        if self._model is None:
            print(f"  🧠 Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"     → Dimension: {self.dimension}")
        return self._model
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a 384-dim vector.
        
        This is what gets called at query time — the user's search string
        is converted to a vector so FAISS can find its nearest neighbors.
        
        Args:
            text: The input string to embed.
        
        Returns:
            numpy array of shape (384,) with dtype float32.
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a batch of text strings into a matrix of vectors.
        
        This is what gets called at index-build time — all document chunks
        are embedded in one batch for efficiency (GPU/CPU batching).
        
        Think of the output as a matrix where:
            - Each row is a document chunk
            - Each column is one of the 384 embedding dimensions
            - The matrix shape is (num_chunks, 384)
        
        Similar to an adjacency matrix, but instead of encoding graph 
        connectivity, each row encodes a point in semantic space.
        
        Args:
            texts:          List of strings to embed.
            batch_size:     Number of texts to process per batch.
            show_progress:  Whether to display a progress bar.
        
        Returns:
            numpy array of shape (len(texts), 384) with dtype float32.
        """
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
