from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List
from urllib.parse import urlparse, unquote
import numpy as np
from sentence_transformers import SentenceTransformer
from app.matching import _extract_pdf_text_from_bytes

def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)

def _get_resume_pdf_bytes(inputs: Dict[str, Any]) -> bytes | None:
    # New format: resume_file = {"filename": "...", "bytes": b"..."}
    f = inputs.get("resume_file")
    if isinstance(f, dict):
        b = f.get("bytes")
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)

    # Old format: resume_file_bytes
    b2 = inputs.get("resume_file_bytes")
    if isinstance(b2, (bytes, bytearray)):
        return bytes(b2)

    return None


def _title_from_linkedin_url(url: str) -> str:
    """
    LinkedIn URLs often look like:
    https://www.linkedin.com/jobs/view/<slug>-<jobid>
    We use the slug as a readable title.
    """
    try:
        path = urlparse(url).path or ""
        path = unquote(path)
        m = re.search(r"/jobs/view/([^/]+)", path)
        if not m:
            return "LinkedIn job"
        slug = m.group(1)
        slug = re.sub(r"-\d+$", "", slug)     # remove trailing job id
        title = slug.replace("-", " ")
        title = re.sub(r"\s+", " ", title).strip()
        return title[:140] if title else "LinkedIn job"
    except Exception:
        return "LinkedIn job"

def _load_mapping(mapping_path: str) -> Any:
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, (dict, list)):
        raise ValueError("jobs_index_mapping.json must be a dict or a list.")
    return mapping


def _mapping_get(mapping: Any, idx: int) -> Any:
    if isinstance(mapping, list):
        return mapping[idx] if 0 <= idx < len(mapping) else None
    if isinstance(mapping, dict):
        return mapping.get(str(idx)) or mapping.get(idx)
    return None

def handle(inputs: Dict[str, Any]) -> Dict[str, Any]:

    resume_pdf_bytes = _get_resume_pdf_bytes(inputs)
    if not resume_pdf_bytes:
        return {"status": "ERROR", "error": "Missing resume_file bytes (expected resume_file dict OR resume_file_bytes)"}

    resume_text = _extract_pdf_text_from_bytes(resume_pdf_bytes)
    if not resume_text.strip():
        return {"status": "ERROR", "error": "Could not extract resume text from PDF (empty)."}

    index_path = inputs.get("jobs_faiss_path") or "./data/linkedin_offers/jobs_index.faiss"
    mapping_path = inputs.get("jobs_mapping_path") or "./data/linkedin_offers/jobs_index_mapping.json"
    top_k = int(inputs.get("top_k") or 10)
    model_name = inputs.get("model_name") or "all-MiniLM-L6-v2"

    if not os.path.exists(index_path):
        return {"status": "ERROR", "error": f"FAISS index not found: {index_path}"}
    if not os.path.exists(mapping_path):
        return {"status": "ERROR", "error": f"Mapping not found: {mapping_path}"}

    # FAISS (import here so requirements can control it)
    try:
        import faiss  # type: ignore
    except Exception:
        return {"status": "ERROR", "error": "faiss is not installed. Install with: pip install faiss-cpu"}

    mapping = _load_mapping(mapping_path)
    index = faiss.read_index(index_path)

    # Embed query
    model = SentenceTransformer(model_name)
    q = model.encode([resume_text], convert_to_numpy=True).astype("float32")
    q = _l2_normalize(q)

    scores, ids = index.search(q, top_k)

    hits: List[Dict[str, Any]] = []
    for rank, (idx, sc) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx < 0:
            continue

        item = _mapping_get(mapping, int(idx))

        hit: Dict[str, Any] = {
            "rank": rank,
            "index_id": int(idx),
            "score": float(sc),
        }

        # mapping stores URL strings
        if isinstance(item, str) and item.startswith("http"):
            hit["url"] = item
            hit["title"] = _title_from_linkedin_url(item)
        else:
            # fallback if mapping contains something unexpected
            hit["url"] = None
            hit["title"] = "LinkedIn job"
            hit["mapping_item"] = item

        hits.append(hit)
    return {
        "status": "OK",
        "mode": "cv_to_jobs_dataset",
        "hits": hits,
        "diagnostics": {
            "index_path": index_path,
            "mapping_path": mapping_path,
            "model_name": model_name,
            "top_k": top_k,
            "note": "Ranking only: mapping contains URLs only (no job text/description available).",
        },
    }
