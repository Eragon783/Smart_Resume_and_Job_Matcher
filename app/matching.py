import json
import numpy as np
import faiss
from io import BytesIO
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdf_extract_text
from ingestion.loaders import clean_text

def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    loaders.py is path-based (read_pdf/read_docx/read_txt). Streamlit provides bytes.
    So we keep this tiny wrapper and still reuse clean_text() from loaders.
    """
    if not pdf_bytes:
        return ""
    raw = pdf_extract_text(BytesIO(pdf_bytes))
    return clean_text(raw)


def _cosine_similarity(model: SentenceTransformer, a: str, b: str) -> float:
    """
    Cosine similarity via dot product of L2-normalized embeddings.
    (We keep it minimal; matching.py covers this logic for FAISS search already.)
    """
    va = model.encode([a], convert_to_numpy=True).astype("float32")
    vb = model.encode([b], convert_to_numpy=True).astype("float32")
    va /= (np.linalg.norm(va, axis=1, keepdims=True) + 1e-12)
    vb /= (np.linalg.norm(vb, axis=1, keepdims=True) + 1e-12)
    return float(np.sum(va * vb))

def _decode_txt_bytes(b: bytes) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")

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

def _get_mapping_item(mapping, idx: int):
    # mapping can be list OR dict of {"0": "...", "1": "..."}
    if isinstance(mapping, list):
        return mapping[idx]
    if isinstance(mapping, dict):
        return mapping.get(str(idx)) or mapping.get(idx)
    return None
