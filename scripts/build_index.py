"""
build_index.py — CLI Entry Point for Building the Knowledge Base

This script orchestrates the full Phase 1 pipeline:
    1. Load all medical documents from data/raw/
    2. Chunk them into ~500-token segments with overlap
    3. Embed each chunk into a 384-dim vector using MiniLM
    4. Build a FAISS HNSW index from the vectors
    5. Save the index + metadata to vectorstore/

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --data-dir path/to/docs --output-dir path/to/output

The entire pipeline runs in-memory and writes results to disk at the end.
Think of it as a batch ETL job: Extract (load) → Transform (chunk + embed) 
→ Load (into FAISS index → disk).
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base.chunker import load_and_chunk_directory
from knowledge_base.embedder import MedicalEmbedder
from knowledge_base.index_builder import FAISSIndexBuilder


def build_knowledge_base(
    data_dir: str = "data/raw",
    output_dir: str = "vectorstore",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> None:
    """
    Run the full knowledge base construction pipeline.
    
    Args:
        data_dir:       Directory containing raw medical documents.
        output_dir:     Directory to save the FAISS index and metadata.
        chunk_size:     Target chunk size in characters.
        chunk_overlap:  Overlap between chunks in characters.
    """
    start_time = time.time()
    
    print("=" * 60)
    print("🏥 Medical Knowledge Base Builder")
    print("=" * 60)
    
    # -----------------------------------------------------------------------
    # Step 1: Load and chunk documents
    # -----------------------------------------------------------------------
    print(f"\n📂 Step 1/3: Loading & chunking documents from '{data_dir}'")
    print("-" * 40)
    
    chunks = load_and_chunk_directory(
        directory=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    if not chunks:
        print("\n❌ No chunks created. Please add documents to the data directory.")
        print(f"   Supported formats: .txt, .pdf")
        print(f"   Expected location: {Path(data_dir).resolve()}")
        sys.exit(1)
    
    # -----------------------------------------------------------------------
    # Step 2: Embed all chunks
    # -----------------------------------------------------------------------
    print(f"\n📐 Step 2/3: Embedding {len(chunks)} chunks with MiniLM")
    print("-" * 40)
    
    embedder = MedicalEmbedder()
    
    # Extract just the text for batch embedding
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_batch(texts)
    
    print(f"  ✅ Embeddings matrix shape: {embeddings.shape}")
    print(f"     → {embeddings.shape[0]} chunks × {embeddings.shape[1]} dimensions")
    
    # -----------------------------------------------------------------------
    # Step 3: Build FAISS HNSW index and save
    # -----------------------------------------------------------------------
    print(f"\n🔧 Step 3/3: Building FAISS HNSW index")
    print("-" * 40)
    
    index_builder = FAISSIndexBuilder(dimension=embedder.dimension)
    
    # Prepare metadata for each chunk (stored alongside the index)
    metadata = [
        {
            "text": chunk.text,
            **chunk.metadata,
        }
        for chunk in chunks
    ]
    
    index_builder.add_vectors(embeddings, metadata)
    
    # Save to disk
    print(f"\n💾 Saving to '{output_dir}'")
    print("-" * 40)
    index_builder.save(output_dir)
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"✅ Knowledge base built successfully!")
    print(f"   📄 Documents processed:  {len(set(c.metadata['source_file'] for c in chunks))}")
    print(f"   🧩 Total chunks:         {len(chunks)}")
    print(f"   📐 Embedding dimension:  {embedder.dimension}")
    print(f"   🔍 HNSW index vectors:   {index_builder.index.ntotal}")
    print(f"   ⏱️  Time elapsed:         {elapsed:.1f}s")
    print(f"{'=' * 60}")
    
    # -----------------------------------------------------------------------
    # Quick sanity check: test a search
    # -----------------------------------------------------------------------
    print(f"\n🧪 Sanity Check: Searching for 'headache symptoms'")
    print("-" * 40)
    
    test_query = "headache symptoms and treatment"
    query_vector = embedder.embed_text(test_query)
    results = index_builder.search(query_vector, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} (score: {result['score']:.4f}):")
        print(f"  Source: {result['source_file']}")
        # Show first 150 chars of the chunk
        preview = result["text"][:150].replace("\n", " ")
        print(f"  Preview: {preview}...")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the medical knowledge base (FAISS HNSW index)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw medical documents (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vectorstore",
        help="Directory to save FAISS index and metadata (default: vectorstore)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Target chunk size in characters (default: 2000, ~500 tokens)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200, ~50 tokens)",
    )
    
    args = parser.parse_args()
    
    build_knowledge_base(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
