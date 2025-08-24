import os
from dotenv import load_dotenv

load_dotenv()  


import json, re, faiss, numpy as np
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI

# --- Config ---
EMBED_MODEL = "text-embedding-3-large"  # good quality; or -small for cheaper
DIM = 3072                               # 3072 for -large, 1536 for -small
SOURCE_PATH = "/Users/rosydai/Desktop/intern/Agentic-RAG-Demo/newsapi.json"
INDEX_PATH = "vector_store/events.faiss"
META_PATH  = "vector_store/events_fused.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def fuse_event(e: Dict[str, Any]) -> str:
    title = _norm_space(e.get("title", {}).get("eng", ""))
    summary = _norm_space(e.get("summary", {}).get("eng", ""))
    city = _norm_space(e.get("location", {}).get("city", ""))
    country = _norm_space(e.get("location", {}).get("country", ""))
    cats = ", ".join(e.get("categories", []) or [])
    parts = [
        "【Event】",
        # f"ID: {e.get('id','')}",
        f"Date: {e.get('eventDate','')}",
        f"Headline: {title}",
        f"Summary: {summary}",
        f"Location: {city}, {country}".strip().strip(", "),
        f"Categories: {cats}" if cats else "",
        # f"ArticleCount: {e.get('totalArticleCount','')}",
        # f"RelevanceScore: {e.get('relevance','')}",
    ]
    text = "\n".join(p for p in parts if p)
    return text

def embed_texts(texts: List[str]) -> np.ndarray:
    # chunk in batches to be safe
    out = []
    for i in range(0, len(texts), 128):
        chunk = texts[i:i+128]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype="float32")

def build_index(json_path: str):
    data = json.load(open(json_path, "r"))
    # If file holds a list directly; if it holds {"events":[...]} adjust accordingly.
    events = data if isinstance(data, list) else data.get("events", [])
    # Dedup on id
    docs, metas =  [], []
    for e in events:
        eid = e.get("id")
        text = fuse_event(e)
        docs.append(text)

    emb = embed_texts(docs)
    # L2-normalize for cosine via inner product
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    index = faiss.index_factory(DIM, "IDMap,Flat")  # simple & local; switch to IVF/HNSW if big
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))  # cosine via dot since we normalized
    ids = np.arange(len(emb), dtype="int64")
    index.add_with_ids(emb, ids)
    faiss.write_index(index, INDEX_PATH)
    json.dump({"metas": docs}, open(META_PATH, "w"))


# --- Example usage ---
# build_index(SOURCE_PATH)


