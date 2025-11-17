"""
build_index.py
Builds FAISS index and saves metadata for semantic search over the Postman Twitter API repo.

Usage:
    python build_index.py --repo_path ./postman-twitter-api --index_path twitter.index --meta_path twitter_meta.json

If repo_path doesn't exist, script will exit with an instruction to clone the repo:
https://github.com/xdevplatform/postman-twitter-api
"""

import os
import argparse
import json
import glob
import math
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -------------------------
# Utilities: chunking logic
# -------------------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

def chunk_text(text, approx_words_per_chunk=250):
    """
    Chunk text by grouping sentences until approx_words_per_chunk reached.
    Returns list of chunk strings.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    current_words = 0
    for s in sentences:
        w = len(s.split())
        if current_words + w > approx_words_per_chunk and current:
            chunks.append(" ".join(current).strip())
            current = [s]
            current_words = w
        else:
            current.append(s)
            current_words += w
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

# -------------------------
# Main build function
# -------------------------
def collect_docs(repo_path):
    # Collect .md, .json, .txt, .yaml, .yml, .html files
    patterns = ["**/*.md", "**/*.json", "**/*.txt", "**/*.yaml", "**/*.yml", "**/*.html"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(repo_path, p), recursive=True))
    # Filter out anything too small
    files = [f for f in files if os.path.getsize(f) > 20]
    files = sorted(files)
    return files

def build_index(repo_path, index_path, meta_path, model_name="all-MiniLM-L6-v2", chunk_words=250, batch_size=64):
    print("Collecting files from", repo_path)
    files = collect_docs(repo_path)
    if not files:
        raise RuntimeError(f"No documentation files found under {repo_path}. Clone the repo first:\nhttps://github.com/xdevplatform/postman-twitter-api")
    print(f"Found {len(files)} files to process.")

    model = SentenceTransformer(model_name)

    metadatas = []  # list of dicts: {source, chunk_id, text}
    embeddings = []

    chunk_counter = 0
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception as e:
            print(f"Skipping {f} (read error):", e)
            continue
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = text.strip()
        if not text:
            continue

        # Chunk the file intelligently
        file_chunks = chunk_text(text, approx_words_per_chunk=chunk_words)
        # If chunking produced nothing (very short file), keep some text
        if not file_chunks:
            file_chunks = [text[:1000]]

        # For each chunk create metadata and compute embeddings in batches
        for cid, chunk in enumerate(file_chunks):
            meta = {
                "source": os.path.relpath(f, repo_path),
                "abs_path": os.path.abspath(f),
                "chunk_id": chunk_counter,
                "chunk_idx_in_file": cid,
                "text": chunk
            }
            metadatas.append(meta)
            chunk_counter += 1

    print(f"Total chunks: {len(metadatas)}. Computing embeddings (model: {model_name})...")

    # Encode all texts in batches to avoid memory spikes
    all_texts = [m["text"] for m in metadatas]
    embeddings = model.encode(all_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    # Normalize embeddings (use cosine search via inner product on normalized vectors)
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index written to {index_path}")

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadatas, mf, indent=2, ensure_ascii=False)
    print(f"Metadata written to {meta_path}")
    print("Build complete.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", default="./postman-twitter-api", help="Path to cloned postman-twitter-api repo")
    parser.add_argument("--index_path", default="twitter.index", help="FAISS index file to write")
    parser.add_argument("--meta_path", default="twitter_meta.json", help="JSON metadata file to write")
    parser.add_argument("--chunk_words", type=int, default=250, help="Approx words per chunk")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    if not os.path.isdir(args.repo_path):
        print(f"Repo path not found: {args.repo_path}")
        print("Please clone the repo first:")
        print("  git clone https://github.com/xdevplatform/postman-twitter-api")
        raise SystemExit(1)

    build_index(args.repo_path, args.index_path, args.meta_path, model_name=args.model, chunk_words=args.chunk_words)
