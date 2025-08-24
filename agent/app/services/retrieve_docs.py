import os
from dotenv import load_dotenv

load_dotenv()  


import json,faiss, numpy as np
from openai import OpenAI
from ..tools.embed_texts import embed_texts

# --- Config ---
EMBED_MODEL = "text-embedding-3-large"  # good quality; or -small for cheaper
DIM = 3072                               # 3072 for -large, 1536 for -small
SOURCE_PATH = "/Users/rosydai/Desktop/intern/Agentic-RAG-Demo/newsapi.json"
META_PATH = "/Users/rosydai/Desktop/intern/Agentic-RAG-Demo/agent/data/vector_store/events_fused.json"
INDEX_PATH  = "/Users/rosydai/Desktop/intern/Agentic-RAG-Demo/agent/data/vector_store/events.faiss"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_docs(query: str, k: int = 10):
    index = faiss.read_index(INDEX_PATH)
    metas = json.load(open(META_PATH))["metas"]
    q_emb = embed_texts([query])
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    sims, ids = index.search(q_emb, k)  # overfetch a bit
    ids = ids[0].tolist(); sims = sims[0].tolist()

    results = []
    for i, s in zip(ids, sims):
        if i < 0: 
            continue
        m = metas[i]
        results.append({"score": float(s), "text": m})
        if len(results) >= k:
            break
    return results

# query = "What is the capital of France?"
# res = retrieve_docs(query, 3)
# for i in res:
#     print(i)