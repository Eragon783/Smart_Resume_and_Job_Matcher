from __future__ import annotations
import json
import numpy as np
import faiss
from io import BytesIO
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdf_extract_text
from ingestion.loaders import clean_text
import json5
import re
import requests
from docx import Document
from typing import Any, Dict, Union, Tuple
import io

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

def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)

def _mapping_get(mapping, idx: int):
    if isinstance(mapping, list):
        return mapping[idx] if 0 <= idx < len(mapping) else None
    if isinstance(mapping, dict):
        return mapping.get(str(idx)) or mapping.get(idx)
    return None

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
    q = _l2_normalize(q)  # keep ONE normalize function

    scores, ids = index.search(q, top_k)

    results = []
    for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
        if i < 0:
            continue
        item = _mapping_get(mapping, int(i))
        results.append({
            "rank": rank,
            "score": float(s),
            "filename": item,
            "index_id": int(i),
        })

    return results

def safe_json_parse(text: str) -> Tuple[Any, str | None]:
    if not text or not isinstance(text, str):
        return None, "empty_text"

    s = text.strip()

    # Removing ```json fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()

    # Extracting the largest JSON object candidate
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_object_found"

    candidate = s[start:end+1].strip()

    # Trying strict JSON first, then json5
    try:
        return json.loads(candidate), None
    except Exception:
        try:
            return json5.loads(candidate), None
        except Exception as e:
            return None, f"extracted_json_parse_failed: {type(e).__name__}"

def sanitize_text_for_llm(text: str) -> str:
    # Removing weird whitespace / control chars
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def smart_trim(text: str, max_chars: int = 12000) -> str:
    # Trimming while keeping head+tail (often best for CVs/jobs)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()


# =======================================================================
# Extract text fromm various file bytes (pdf, docx, txt)

def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    loaders.py is path-based (read_pdf/read_docx/read_txt). Streamlit provides bytes.
    So we keep this tiny wrapper and still reuse clean_text() from loaders.
    """
    if not pdf_bytes:
        return ""
    raw = pdf_extract_text(BytesIO(pdf_bytes))
    return clean_text(raw)

def _decode_txt_bytes(b: bytes) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")

def extract_text_from_file(file_or_name, b: bytes = None) -> str:
    if isinstance(file_or_name, dict):
        filename = (file_or_name.get("filename") or "")
        data = file_or_name.get("bytes") or b""
    else:
        filename = file_or_name or ""
        data = b or b""

    ext = (filename.split(".")[-1] if "." in filename else "").lower()

    if ext == "pdf":
        return _extract_pdf_text_from_bytes(bytes(data))

    if ext == "txt":
        return _decode_txt_bytes(bytes(data)).strip()

    if ext == "docx":
        doc = Document(io.BytesIO(bytes(data)))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    # fallback
    return _decode_txt_bytes(bytes(data)).strip()
