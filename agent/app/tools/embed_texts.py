import os
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from typing import List
from openai import OpenAI

# --- Config ---
EMBED_MODEL = "text-embedding-3-large"  # good quality; or -small for cheaper
DIM = 3072                               # 3072 for -large, 1536 for -small
SOURCE_PATH = "/Users/rosydai/Desktop/intern/Agentic-RAG-Demo/newsapi.json"
INDEX_PATH = "vector_store/events.faiss"
META_PATH  = "vector_store/events_fused.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: List[str]) -> np.ndarray:
    # chunk in batches to be safe
    out = []
    for i in range(0, len(texts), 128):
        chunk = texts[i:i+128]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype="float32")