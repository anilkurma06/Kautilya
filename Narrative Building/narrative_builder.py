#!/usr/bin/env python3
"""
narrative_builder.py

Usage:
  python narrative_builder.py --topic "Jubilee Hills elections" --dataset ./news_dataset.json

This script:
 - Loads a JSON news dataset (array or JSONL)
 - Filters articles with `source_rating` > min_rating (default 8)
 - Finds articles relevant to the provided topic using embeddings
 - Produces a narrative JSON with summary, timeline, clusters and graph

Optional: set OPENAI_API_KEY to enable better generative summaries (fallback to extractive otherwise).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from math import ceil

import numpy as np
from dateutil import parser as dateparser
from sklearn.cluster import KMeans
import networkx as nx

# Try to import sentence-transformers, but provide a TF-IDF fallback if it's unavailable or slow to load.
HAVE_SBERT = False
EMBED_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SBERT = True
except Exception:
    HAVE_SBERT = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")
TFIDF_VECTORIZER = None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback to JSONL parsing
            objs = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except json.JSONDecodeError:
                    continue
            return objs
        
        # If data is a dict with an 'items' key, extract it
        if isinstance(data, dict) and 'items' in data:
            data = data['items']
        
        # Ensure we have a list of dicts
        if isinstance(data, list):
            # Filter out non-dict items
            data = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            # Single dict, wrap in list
            data = [data]
        else:
            return []
        
        return data


def pick_text_field(article):
    # Try several common fields for article text
    for key in ("story", "content", "body", "article", "summary", "description", "text"):
        v = article.get(key)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: join other string fields
    parts = []
    for k, v in article.items():
        if isinstance(v, str) and len(v) > 30:
            parts.append(v.strip())
    return "\n\n".join(parts) if parts else ""


def parse_date(article):
    for key in ("date", "published", "published_at", "timestamp"):
        v = article.get(key)
        if not v:
            continue
        try:
            return dateparser.parse(v)
        except Exception:
            continue
    return None


def filter_by_rating(articles, min_rating=8):
    out = []
    for a in articles:
        r = a.get("source_rating")
        try:
            if r is None:
                continue
            if float(r) > min_rating:
                out.append(a)
        except Exception:
            continue
    return out


def ensure_model():
    global EMBED_MODEL, TFIDF_VECTORIZER
    if HAVE_SBERT:
        try:
            EMBED_MODEL = SentenceTransformer(MODEL_NAME)
            return
        except Exception:
            # fall back to TF-IDF
            pass
    # TF-IDF fallback (fast, lightweight)
    TFIDF_VECTORIZER = TfidfVectorizer(max_features=16384, stop_words='english')
    EMBED_MODEL = None


def embed_texts(texts):
    if HAVE_SBERT and EMBED_MODEL is not None:
        emb = EMBED_MODEL.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return np.array(emb)
    # TF-IDF fallback
    global TFIDF_VECTORIZER
    return TFIDF_VECTORIZER.transform(texts).toarray()


def get_relevant_articles(articles, topic, top_k=200):
    texts = [pick_text_field(a) for a in articles]
    ids = list(range(len(articles)))
    if HAVE_SBERT and EMBED_MODEL is not None:
        topic_emb = EMBED_MODEL.encode(topic, convert_to_tensor=False)
        corpus_emb = embed_texts(texts)
    else:
        # fit TF-IDF on corpus + topic
        global TFIDF_VECTORIZER
        TFIDF_VECTORIZER.fit(texts + [topic])
        corpus_emb = TFIDF_VECTORIZER.transform(texts).toarray()
        topic_emb = TFIDF_VECTORIZER.transform([topic]).toarray()[0]
    sims = cosine_similarity([topic_emb], corpus_emb)[0]
    ranked = sorted(list(zip(ids, sims)), key=lambda x: x[1], reverse=True)
    selected = [articles[i] for i, s in ranked[:top_k] if texts[i].strip()]
    scores = [float(s) for i, s in ranked[:top_k] if texts[i].strip()]
    return selected, scores


def sentence_tokenize(text):
    import re
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]


def build_summary_extractive(articles, topic, min_sents=5, max_sents=10):
    texts = [pick_text_field(a) for a in articles]
    combined = "\n\n".join(texts)
    sentences = sentence_tokenize(combined)
    if not sentences:
        return ""
    if HAVE_SBERT and EMBED_MODEL is not None:
        sent_embs = EMBED_MODEL.encode(sentences, convert_to_tensor=False, show_progress_bar=False)
        topic_emb = EMBED_MODEL.encode(topic, convert_to_tensor=False)
    else:
        global TFIDF_VECTORIZER
        TFIDF_VECTORIZER.fit(sentences + [topic])
        sent_embs = TFIDF_VECTORIZER.transform(sentences).toarray()
        topic_emb = TFIDF_VECTORIZER.transform([topic]).toarray()[0]
    sims = cosine_similarity([topic_emb], sent_embs)[0]
    ranked_idx = list(np.argsort(-sims))
    chosen = []
    for idx in ranked_idx:
        if len(chosen) >= max_sents:
            break
        s = sentences[int(idx)]
        if any(s in c or c in s for c in chosen):
            continue
        chosen.append(s)
    if len(chosen) < min_sents:
        for idx in ranked_idx:
            s = sentences[int(idx)]
            if s not in chosen:
                chosen.append(s)
            if len(chosen) >= min_sents:
                break
    return " ".join(chosen[:max_sents])


def representative_sentence_for_article(article, topic_emb):
    text = pick_text_field(article)
    sents = sentence_tokenize(text)
    if not sents:
        return ""
    if HAVE_SBERT and EMBED_MODEL is not None:
        emb = EMBED_MODEL.encode(sents, convert_to_tensor=False, show_progress_bar=False)
    else:
        global TFIDF_VECTORIZER
        TFIDF_VECTORIZER.fit(sents + ["topic"])
        emb = TFIDF_VECTORIZER.transform(sents).toarray()
    sims = cosine_similarity([topic_emb], emb)[0]
    idx = int(np.argmax(sims))
    return sents[idx]


def cluster_articles(articles, num_clusters=None):
    texts = [pick_text_field(a) for a in articles]
    if HAVE_SBERT and EMBED_MODEL is not None:
        embs = EMBED_MODEL.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    else:
        embs = TFIDF_VECTORIZER.transform(texts).toarray()
    n = len(articles)
    if n <= 2:
        labels = [0] * n
        return labels
    if num_clusters is None:
        num_clusters = min(10, max(2, n // 10))
    km = KMeans(n_clusters=num_clusters, random_state=0)
    labels = km.fit_predict(embs)
    return labels


def sentiment_scores(texts):
    if not HAS_TRANSFORMERS:
        return [0.0] * len(texts)
    try:
        classifier = pipeline("sentiment-analysis", truncation=True)
    except Exception:
        return [0.0] * len(texts)
    outs = classifier(texts)
    scores = []
    for o in outs:
        lab = o.get("label", "NEUTRAL")
        sc = float(o.get("score", 0.0))
        if lab.upper().startswith("NEG"):
            scores.append(-sc)
        else:
            scores.append(sc)
    return scores


def build_graph(articles):
    texts = [pick_text_field(a) for a in articles]
    if HAVE_SBERT and EMBED_MODEL is not None:
        emb = EMBED_MODEL.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    else:
        emb = TFIDF_VECTORIZER.transform(texts).toarray()
    simmat = cosine_similarity(emb, emb)
    dates = [parse_date(a) for a in articles]
    sentiments = sentiment_scores(texts)
    G = nx.DiGraph()
    for i, a in enumerate(articles):
        G.add_node(i, headline=a.get('headline') or a.get('title') or '', url=a.get('url'))
    for i in range(len(articles)):
        for j in range(len(articles)):
            if i == j:
                continue
            s = float(simmat[i, j])
            if s < 0.55:
                continue
            di = dates[i]
            dj = dates[j]
            # Normalize dates to naive UTC for comparison
            if di:
                di = di.replace(tzinfo=None) if di.tzinfo else di
            if dj:
                dj = dj.replace(tzinfo=None) if dj.tzinfo else dj
            
            if di and dj and dj > di and s > 0.7:
                G.add_edge(i, j, relation='builds_on', weight=s)
            elif s > 0.7 and ((di and dj and dj < di) or not di or not dj):
                G.add_edge(i, j, relation='adds_context', weight=s)
            elif 0.6 <= s <= 0.7:
                G.add_edge(i, j, relation='adds_context', weight=s)
            if abs(sentiments[i]) > 0 and abs(sentiments[j]) > 0 and (sentiments[i] * sentiments[j] < 0) and s > 0.6:
                G.add_edge(i, j, relation='contradicts', weight=s)
            later = None
            if di and dj:
                later = j if dj > di else (i if di > dj else None)
            if later == j and s > 0.65:
                txt = texts[j].lower()
                if any(k in txt for k in ('escalat', 'attack', 'clash', 'violence', 'intensif')):
                    G.add_edge(i, j, relation='escalates', weight=s)
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"source": int(u), "target": int(v), "relation": d.get('relation'), "weight": float(d.get('weight', 0))})
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({"id": int(n), "headline": d.get('headline', ''), "url": d.get('url', '')})
    return {"nodes": nodes, "edges": edges}


def build_timeline(articles, topic_emb):
    entries = []
    for a in articles:
        d = parse_date(a)
        rep = representative_sentence_for_article(a, topic_emb)
        entries.append({
            "date": d.isoformat() if d else None,
            "headline": a.get('headline') or a.get('title') or '',
            "url": a.get('url'),
            "why_it_matters": rep
        })
    entries = sorted(entries, key=lambda x: x['date'] or '')
    return entries


def main():
    parser = argparse.ArgumentParser(description='Build a narrative from a news dataset')
    parser.add_argument('--topic', required=True, help='Topic to build narrative for')
    parser.add_argument('--dataset', help='Path to JSON dataset (array or JSONL). Default: news_dataset.json or sample_news.json')
    parser.add_argument('--min_rating', type=float, default=8.0, help='Minimum source_rating (exclusive)')
    parser.add_argument('--top_k', type=int, default=200, help='Number of top relevant articles to consider')
    parser.add_argument('--output', help='Path to output JSON file (defaults to stdout)')
    args = parser.parse_args()

    # Determine dataset path
    dataset_path = args.dataset
    if not dataset_path:
        # Try default locations
        if os.path.exists('news_dataset.json'):
            dataset_path = 'news_dataset.json'
        elif os.path.exists('sample_news.json'):
            dataset_path = 'sample_news.json'
        else:
            print("Error: No dataset found. Please specify --dataset or provide news_dataset.json or sample_news.json", file=sys.stderr)
            sys.exit(2)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading dataset from {dataset_path}...", file=sys.stderr)
    articles = load_json(dataset_path)
    print(f"Loaded {len(articles)} articles", file=sys.stderr)

    print("Filtering by source_rating...", file=sys.stderr)
    filtered = filter_by_rating(articles, min_rating=args.min_rating)
    print(f"{len(filtered)} articles after rating filter", file=sys.stderr)

    if not filtered:
        print(json.dumps({"error": "no_articles_after_filter"}))
        sys.exit(0)

    ensure_model()

    print("Finding relevant articles...", file=sys.stderr)
    relevant, scores = get_relevant_articles(filtered, args.topic, top_k=args.top_k)
    print(f"Selected {len(relevant)} relevant articles", file=sys.stderr)

    # Create topic embedding (or use TF-IDF representation)
    if HAVE_SBERT and EMBED_MODEL is not None:
        topic_emb = EMBED_MODEL.encode(args.topic, convert_to_tensor=False)
    else:
        global TFIDF_VECTORIZER
        topic_emb = TFIDF_VECTORIZER.transform([args.topic]).toarray()[0]

    print("Building extractive narrative summary...", file=sys.stderr)
    summary = build_summary_extractive(relevant, args.topic)

    timeline = build_timeline(relevant, topic_emb)

    print("Clustering articles...", file=sys.stderr)
    labels = cluster_articles(relevant)
    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        a = relevant[idx]
        clusters[int(lab)].append({
            "id": idx,
            "headline": a.get('headline') or a.get('title') or '',
            "url": a.get('url')
        })
    clusters_out = [ {"cluster_id": k, "articles": v} for k, v in clusters.items()]

    print("Building narrative graph...", file=sys.stderr)
    graph = build_graph(relevant)

    out = {
        "narrative_summary": summary,
        "timeline": timeline,
        "clusters": clusters_out,
        "graph": graph
    }

    j = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(j)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(j)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
narrative_builder.py

Usage:
    python narrative_builder.py --topic "Israel-Iran conflict" --news_path ./news.json --top_k 100

Description:
- Loads news JSON dataset (expects a list of article objects with at least: "date", "headline", "content", "url", "source_rating")
- Filters by source_rating > 8
- Retrieves top-K relevant articles to --topic using BM25
- Produces:
    - narrative_summary: 5-10 sentences
    - timeline: chronological list of relevant articles (date, headline, url, why_it_matters)
    - clusters: groups of similar articles
    - graph: nodes & edges with relation types (heuristic)
- Prints final JSON to stdout
"""

import argparse
import json
import math
import re
from collections import defaultdict
from dateutil import parser as dateparser
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# -------------------------
# Helpers
# -------------------------
SENT_END_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text):
    if not text:
        return []
    s = re.sub(r'\s+', ' ', text.strip())
    return [seg.strip() for seg in SENT_END_RE.split(s) if seg.strip()]

def safe_get(article, key, default=""):
    return article.get(key) or default

def parse_date(dstr):
    # returns naive datetime or None
    try:
        return dateparser.parse(dstr)
    except Exception:
        return None

# -------------------------
# Retrieval (BM25)
# -------------------------
def retrieve_relevant_articles(data, topic, top_k=100):
    # Build corpus for BM25 using content+headline
    tokenized = []
    corpus_texts = []
    for a in data:
        text = (safe_get(a, "headline", "") + " " + safe_get(a, "content", "")).lower()
        corpus_texts.append(text)
        # simple tokenization (split on whitespace, keep alphanum)
        tokens = re.findall(r"\w+", text)
        tokenized.append(tokens)
    bm25 = BM25Okapi(tokenized)
    query_tokens = re.findall(r"\w+", topic.lower())
    top_indices = bm25.get_top_n(query_tokens, list(range(len(corpus_texts))), n=top_k)
    # top_indices are actual indices; convert to articles
    results = [data[i] for i in top_indices]
    return results
if __name__ == '__main__':
    main()

