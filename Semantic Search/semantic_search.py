"""
semantic_search.py

Load the FAISS index and metadata, perform semantic search for a user query, and print JSON to stdout.

Usage:
    python semantic_search.py --query "How do I fetch tweets with expansions?" --k 5
    python semantic_search.py --query "ratelimit" --k 10 --index_path twitter.index --meta_path twitter_meta.json

If index/meta are missing, pass --rebuild to trigger build_index.py (requires the repo be cloned).
"""

import argparse
import json
import os
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import subprocess

DEFAULT_INDEX = "twitter.index"
DEFAULT_META = "twitter_meta.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

def ensure_index(index_path, meta_path, repo_path, model):
    if os.path.exists(index_path) and os.path.exists(meta_path):
        return True

    print("Index or metadata not found.")

    # If repo is missing, attempt to clone it first
    if not os.path.isdir(repo_path):
        print(f"Repository path '{repo_path}' not found. Attempting to clone from GitHub...")
        repo_url = "https://github.com/xdevplatform/postman-twitter-api"
        try:
            subprocess.check_call(["git", "clone", repo_url, repo_path])
        except Exception as e:
            print("Failed to clone repository:", e)
            return False

    print("Building index now (this may take a few minutes)...")
    cmd = [sys.executable, "build_index.py", "--repo_path", repo_path, "--index_path", index_path, "--meta_path", meta_path, "--model", model]
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as e:
        print("Automatic build failed:", e)
        return False

def search(index_path, meta_path, query, k=5, model_name=DEFAULT_MODEL):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata not found. Build it first or use --rebuild.")

    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    results = []
    for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
        if idx < 0 or idx >= len(metadatas):
            continue
        meta = metadatas[idx]
        # include a short snippet (first 400 chars)
        snippet = meta["text"][:400]
        results.append({
            "rank": rank,
            "score": float(score),
            "source": meta.get("source"),
            "abs_path": meta.get("abs_path"),
            "chunk_id": meta.get("chunk_id"),
            "chunk_idx_in_file": meta.get("chunk_idx_in_file"),
            "snippet": snippet,
            "text": meta["text"]  # full chunk text
        })
    output = {
        "query": query,
        "k": k,
        "num_results": len(results),
        "results": results
    }
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=5, help="Top-k results")
    parser.add_argument("--index_path", default=DEFAULT_INDEX)
    parser.add_argument("--meta_path", default=DEFAULT_META)
    parser.add_argument("--repo_path", default="./postman-twitter-api", help="Path to cloned repo (used if --rebuild)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index before searching (calls build_index.py)")
    args = parser.parse_args()

    if args.rebuild:
        built = ensure_index(args.index_path, args.meta_path, args.repo_path, args.model)
        if not built:
            print("Could not build index. Exiting.")
            raise SystemExit(1)

    try:
        out = search(args.index_path, args.meta_path, args.query, k=args.k, model_name=args.model)
    except FileNotFoundError:
        # try to auto-build once
        ok = ensure_index(args.index_path, args.meta_path, args.repo_path, args.model)
        if not ok:
            print("Index missing and automatic build failed. Clone the repo and run build_index.py manually.")
            raise SystemExit(1)
        out = search(args.index_path, args.meta_path, args.query, k=args.k, model_name=args.model)

    # Print JSON to stdout using UTF-8 to avoid encoding errors when redirected
    out_json = json.dumps(out, indent=2, ensure_ascii=False)
    try:
        # Write bytes to stdout buffer so Windows consoles with limited encodings won't fail
        sys.stdout.buffer.write(out_json.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
    except Exception:
        # Fallback to print (may raise UnicodeEncodeError in some consoles)
        print(out_json)
