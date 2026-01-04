import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-12, None)

def search_index(
    query_text: str,
    index_path: str,
    mapping_path: str,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
):
    import faiss

    # load index + mapping
    index = faiss.read_index(index_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # embed query
    model = SentenceTransformer(model_name)
    q = model.encode([query_text], convert_to_numpy=True).astype("float32")
    q = _normalize(q)

    scores, ids = index.search(q, top_k)

    results = []
    for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
        if i < 0:
            continue
        results.append({
            "rank": rank,
            "score": float(s),
            "filename": mapping[i],
        })

    return results

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
