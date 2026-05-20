"""
index_builder.py — FAISS HNSW Index Construction

Builds a FAISS IndexHNSWFlat for approximate nearest-neighbor search over
medical document embeddings.

HNSW (Hierarchical Navigable Small World) Internals:
    
    Think of HNSW as a "skip list for high-dimensional vectors":
    
    1. Layer 0 (bottom):  Contains ALL vectors, connected to their nearest
                          neighbors forming a navigable graph.
    2. Layer 1..L (upper): Contain progressively FEWER vectors (sampled 
                          probabilistically), with longer-range connections.
    
    Search traverses top-down:
        - Start at the top layer (sparse, long jumps → coarse navigation)
        - Descend layer by layer (denser, short jumps → fine-grained)
        - At layer 0, perform a greedy beam search among local neighbors
    
    Time complexity: O(log N) for search (vs O(N) for brute-force)
    Space complexity: O(N * M) where M = edges per node
    
    The key parameter is M (number of connections per node):
        - Higher M → better recall but more memory and slower builds
        - M=32 is a solid default for most use cases
    
    This is conceptually similar to how skip lists achieve O(log N) by 
    maintaining multiple layers of linked lists with express lanes.

Usage:
    builder = FAISSIndexBuilder(dimension=384)
    builder.add_vectors(embeddings_matrix)
    builder.save("vectorstore/medical_index.faiss")
"""

import os
import pickle

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# FAISS HNSW Index Builder
# ---------------------------------------------------------------------------

class FAISSIndexBuilder:
    """
    Builds and manages a FAISS HNSW index for medical document retrieval.
    
    Attributes:
        dimension:  Vector dimension (384 for MiniLM-L6-v2).
        m:          Number of connections per node in the HNSW graph.
                    Higher = better recall, more memory. Default: 32.
        ef_construction: Size of the dynamic candidate list during build.
                    Higher = better index quality, slower build. Default: 200.
        ef_search:  Size of the dynamic candidate list during search.
                    Higher = better recall, slower search. Default: 128.
        index:      The FAISS IndexHNSWFlat instance.
    """
    
    def __init__(
        self,
        dimension: int = 384,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ):
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Create the HNSW index
        # IndexHNSWFlat = HNSW graph structure + flat (exact) distance calc
        self.index = faiss.IndexHNSWFlat(dimension, m)
        
        # Set construction-time beam width
        # (how many candidates to consider when inserting a new node)
        self.index.hnsw.efConstruction = ef_construction
        
        # Set search-time beam width
        # (how many candidates to explore during query)
        self.index.hnsw.efSearch = ef_search
        
        # Storage for chunk metadata (text + source info)
        self.chunks_metadata: list[dict] = []
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Add vectors and their associated metadata to the index.
        
        The vectors and metadata must be aligned — vectors[i] corresponds 
        to metadata[i]. Think of it like parallel arrays where the FAISS 
        index stores vectors by integer ID, and we maintain a separate 
        lookup table (metadata list) indexed by the same ID.
        
        Args:
            vectors:    numpy array of shape (N, dimension), dtype float32.
            metadata:   List of dicts, one per vector. Each dict should 
                        contain at minimum: {text, source_file, chunk_index}.
        
        Raises:
            ValueError: If vectors and metadata lengths don't match.
            ValueError: If vector dimension doesn't match index dimension.
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"Vector count ({len(vectors)}) != metadata count ({len(metadata)})"
            )
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) != "
                f"index dimension ({self.dimension})"
            )
        
        # Ensure float32 (FAISS requirement)
        vectors = vectors.astype(np.float32)
        
        self.index.add(vectors)
        self.chunks_metadata.extend(metadata)
        
        print(f"  📊 Added {len(vectors)} vectors to HNSW index")
        print(f"     → Total vectors in index: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Search the index for the top-K nearest neighbors.
        
        Args:
            query_vector:   numpy array of shape (dimension,) or (1, dimension).
            top_k:          Number of nearest neighbors to return.
        
        Returns:
            List of dicts, each containing:
                - text: The chunk text
                - source_file: Original document
                - chunk_index: Position in the original document
                - score: L2 distance (lower = more similar)
        """
        # Reshape to (1, dimension) if needed — FAISS expects a batch
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype(np.float32)
        
        # distances: (1, top_k) array of L2 distances
        # indices:   (1, top_k) array of vector IDs
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for "no result" when index is too small
                continue
            
            result = {
                **self.chunks_metadata[idx],
                "score": float(dist),
            }
            results.append(result)
        
        return results
    
    def save(self, directory: str) -> None:
        """
        Save the FAISS index and chunk metadata to disk.
        
        Creates two files:
            - medical_index.faiss  → the HNSW index (FAISS binary format)
            - chunks_metadata.pkl  → chunk text + metadata (Python pickle)
        
        Args:
            directory: Path to the output directory.
        """
        os.makedirs(directory, exist_ok=True)
        
        index_path = os.path.join(directory, "medical_index.faiss")
        metadata_path = os.path.join(directory, "chunks_metadata.pkl")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"  💾 FAISS index saved: {index_path}")
        
        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks_metadata, f)
        print(f"  💾 Metadata saved:    {metadata_path}")
    
    @classmethod
    def load(cls, directory: str) -> "FAISSIndexBuilder":
        """
        Load a previously saved FAISS index and metadata from disk.
        
        This is a class method — it creates a new FAISSIndexBuilder instance
        with the loaded index. Used at query time (Phase 3) when the agent's
        retrieval tool needs to search the pre-built index.
        
        Args:
            directory: Path to the directory containing saved files.
        
        Returns:
            A FAISSIndexBuilder instance with the loaded index and metadata.
        
        Raises:
            FileNotFoundError: If index or metadata files are missing.
        """
        index_path = os.path.join(directory, "medical_index.faiss")
        metadata_path = os.path.join(directory, "chunks_metadata.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Create instance and restore state
        builder = cls(dimension=index.d)
        builder.index = index
        builder.chunks_metadata = metadata
        
        print(f"  ✅ Loaded FAISS index: {index.ntotal} vectors, dim={index.d}")
        return builder
