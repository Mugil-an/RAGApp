"""PDF -> chunks -> embeddings -> qdrant upsert utility using Google AI Studio.

Usage:
  - Set GOOGLE_API_KEY in your environment
  - Run as script: python data_loader.py --paths file1.pdf file2.pdf

This script will:
  1. extract text from PDFs using LlamaIndex PDFReader,
  2. split into overlapping chunks using SentenceSplitter,
  3. compute embeddings using Google AI Studio (Gemini),
  4. upsert to Qdrant via the QdrantStora in vector_db.py
"""
from __future__ import annotations

import os
import uuid
import logging
import argparse
import datetime
from typing import List

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from llama_index.readers.file import PDFReader
    from llama_index.core.node_parser import SentenceSplitter
except ImportError:
    PDFReader = None
    SentenceSplitter = None

from dotenv import load_dotenv
from vector_db import QdrantStorage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google AI Studio configuration
EMBED_MODEL = "models/text-embedding-004"  # Google's latest embedding model
EMBED_DIM = 768  # Dimension for text-embedding-004

# Initialize Google AI
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment")

# Text splitter configuration
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200) if SentenceSplitter else None


def load_and_chunk_pdf(path: str) -> List[str]:
    """Load PDF and split into chunks using LlamaIndex.
    
    Same structure as your example code but with error handling.
    """
    if PDFReader is None or SentenceSplitter is None:
        raise RuntimeError("LlamaIndex not installed. Run: pip install llama-index")
    
    logger.info("Loading PDF: %s", path)
    
    # Load PDF using LlamaIndex PDFReader
    reader = PDFReader()
    docs = reader.load_data(file=path)
    
    # Extract text from documents
    texts = [d.text for d in docs if getattr(d, "text", None)]
    
    # Split into chunks
    chunks = []
    for text in texts:
        if text.strip():
            chunks.extend(splitter.split_text(text))
    
    logger.info("Generated %d chunks from %s", len(chunks), path)
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using Google AI Studio.
    
    Same structure as your example code but using Google AI Studio API.
    """
    if not texts:
        return []
    
    if genai is None:
        raise RuntimeError("Google GenerativeAI not installed. Run: pip install google-generativeai")
    
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Google AI Studio embeddings")
    
    logger.info("Embedding %d texts using %s", len(texts), EMBED_MODEL)
    
    try:
        # Use Google AI Studio embedding API
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=texts,
            task_type="retrieval_document"
        )
        
        # Extract embeddings from response
        embeddings = response['embedding'] if isinstance(response['embedding'][0], list) else [response['embedding']]
        
        logger.info("Successfully generated %d embeddings", len(embeddings))
        return embeddings
        
    except Exception as e:
        logger.exception("Google AI Studio embedding failed: %s", e)
        raise RuntimeError(f"Google AI Studio embedding failed: {e}")


def batch_texts(texts: List[str], batch_size: int = 64) -> List[List[str]]:
    """Split texts into batches for embedding."""
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i + batch_size])
    return batches


def load_and_index(paths: List[str], qdrant_url: str = None, collection: str = "docs", batch_size: int = 64):
    """Main entry: loads PDF files, creates chunks, computes embeddings, and upserts to Qdrant.
    
    Uses the same clean structure as your example code.
    Returns a summary dict.
    """
    if not paths:
        raise ValueError("No PDF paths provided")

    # Initialize Qdrant storage with correct embedding dimension
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant = QdrantStorage(url=qdrant_url, collection=collection, dim=EMBED_DIM)

    total_chunks = 0
    total_upserted = 0
    errors = []

    for pdf_path in paths:
        logger.info("Processing %s", pdf_path)
        
        try:
            # Load and chunk PDF (like your example)
            chunks = load_and_chunk_pdf(pdf_path)
            total_chunks += len(chunks)
            
            if not chunks:
                logger.warning("No chunks extracted from %s", pdf_path)
                continue

            # Process chunks in batches
            all_ids = []
            all_vectors = []
            all_payloads = []
            
            for batch_chunks in batch_texts(chunks, batch_size):
                try:
                    # Embed texts (like your example)
                    embeddings = embed_texts(batch_chunks)
                    
                    # Prepare data for Qdrant
                    for i, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                        chunk_id = str(uuid.uuid4())
                        payload = {
                            "text": chunk_text,
                            "source": os.path.abspath(pdf_path),
                            "chunk_index": len(all_ids) + i,
                            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                        }
                        
                        all_ids.append(chunk_id)
                        all_vectors.append(embedding)
                        all_payloads.append(payload)
                        
                except Exception as e:
                    logger.exception("Embedding failed for batch in %s", pdf_path)
                    errors.append({"path": pdf_path, "error": f"embed: {e}"})
                    continue
            
            # Upsert to Qdrant
            if all_vectors:
                try:
                    qdrant.upsert(all_ids, all_vectors, all_payloads)
                    total_upserted += len(all_vectors)
                    logger.info("Upserted %d chunks from %s", len(all_vectors), pdf_path)
                except Exception as e:
                    logger.exception("Failed to upsert to Qdrant for %s", pdf_path)
                    errors.append({"path": pdf_path, "error": "qdrant upsert failed"})
                    
        except Exception as e:
            logger.exception("Failed to process %s", pdf_path)
            errors.append({"path": pdf_path, "error": str(e)})

    return {
        "processed_files": len(paths),
        "total_chunks": total_chunks,
        "total_upserted": total_upserted,
        "errors": errors,
    }


def _gather_paths(paths_or_dir: List[str]) -> List[str]:
    out = []
    for p in paths_or_dir:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        out.append(os.path.join(root, f))
        else:
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Index PDFs into Qdrant")
    parser.add_argument("--paths", nargs="+", required=True, help="PDF file(s) or directory(ies) to index")
    parser.add_argument("--collection", default="docs")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--qdrant-url", default=None)

    args = parser.parse_args()
    paths = _gather_paths(args.paths)
    summary = load_and_index(paths, qdrant_url=args.qdrant_url, collection=args.collection,
                            batch_size=args.batch_size)
    print(summary)


if __name__ == "__main__":
    main()
