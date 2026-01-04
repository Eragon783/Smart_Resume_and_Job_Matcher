# app/pipeline.py
from __future__ import annotations

from typing import Dict, Any, List
import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.loaders import clean_text  # reusing your existing ingestion code
from agents.explainer_agent import explain_match_with_llm, build_llm_client  # reusing your existing agent


# ---------------------------
# Helpers (runtime only)
# ---------------------------

def _decode_txt_bytes(b: bytes) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")


def _encode_normalized(model: SentenceTransformer, text: str) -> np.ndarray:
    """
    Embed + L2-normalize.
    This matches the cosine-similarity setup used with IndexFlatIP.
    """
    emb = model.encode([text], convert_to_numpy=True).astype("float32")  # (1, d)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb  # (1, d)


def _load_faiss_index(index_path: str, mapping_path: str):
    import faiss  # faiss-cpu
    index = faiss.read_index(index_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)  # list[str] pos -> filename
    return index, mapping


def _faiss_search(index, mapping: List[str], query_vec: np.ndarray, top_k: int):
    scores, idxs = index.search(query_vec, top_k)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    hits = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if 0 <= i < len(mapping):
            hits.append({"rank": rank, "filename": mapping[i], "score": float(s)})
    return hits


def _load_resume_text(resume_txt_folder: str, filename: str, max_chars: int = 7000) -> str:
    path = os.path.join(resume_txt_folder, filename)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return clean_text(txt)[:max_chars]


# ---------------------------
# Pipeline router
# ---------------------------

def run(mode: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if mode == "job_to_cvs_dataset":
        return _run_job_to_cvs_dataset(inputs)

    # Other modes will be implemented later, one by one
    return {
        "status": "NOT_IMPLEMENTED_YET",
        "mode": mode,
        "inputs_received": list(inputs.keys()),
    }


# ---------------------------
# MODE: Find best resumes in dataset for one job offer (FAISS)
# ---------------------------

def _run_job_to_cvs_dataset(inputs: Dict[str, Any]) -> Dict[str, Any]:
    job_offer = inputs.get("job_offer_file")
    if not job_offer or not job_offer.get("bytes"):
        return {"status": "ERROR", "error": "Missing job_offer_file (TXT)"}

    job_text = clean_text(_decode_txt_bytes(job_offer["bytes"]))
    if not job_text.strip():
        return {"status": "ERROR", "error": "Job offer text is empty/unreadable"}

    # Defaults (change here if your repo paths differ)
    faiss_index_path = inputs.get("resume_faiss_path") or "./data/resume_index.faiss"
    mapping_path = inputs.get("resume_mapping_path") or "./data/resume_index_mapping.json"
    resume_txt_folder = inputs.get("resume_txt_folder") or "./data/resume_extract_text"

    # Pre-flight checks with explicit errors
    if not os.path.exists(faiss_index_path):
        return {
            "status": "ERROR",
            "error": "FAISS index file not found",
            "expected_path": faiss_index_path,
        }
    if not os.path.exists(mapping_path):
        return {
            "status": "ERROR",
            "error": "Mapping JSON file not found",
            "expected_path": mapping_path,
        }

    # Loading prebuilt index (NO rebuilding / NO dataset embedding)
    index, mapping = _load_faiss_index(faiss_index_path, mapping_path)

    # Diagnostics: index + mapping sanity
    faiss_ntotal = int(getattr(index, "ntotal", -1))
    mapping_len = len(mapping)
    resume_txt_folder_exists = os.path.isdir(resume_txt_folder)

    # Embedding only query
    model_name = inputs.get("model_name") or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    query_vec = _encode_normalized(model, job_text)

    top_k = int(inputs.get("top_k") or 10)
    hits = _faiss_search(index, mapping, query_vec, top_k=top_k)

    # Optional explanations (top N only)
    add_explanations = bool(inputs.get("add_explanations") or False)
    explain_top_n = int(inputs.get("explain_top_n") or inputs.get("explain_top_n") or 3)

    # If explanations are requested but folder is missing, keep going but report it
    explanations_errors = []
    if add_explanations and not resume_txt_folder_exists:
        explanations_errors.append(
            f"resume_txt_folder does not exist: {resume_txt_folder}. "
            f"Explanations require loading resume .txt files by filename."
        )

    if add_explanations and hits and resume_txt_folder_exists:
        client = build_llm_client()
        for h in hits[:min(explain_top_n, len(hits))]:
            try:
                resume_text = _load_resume_text(resume_txt_folder, h["filename"])
                h["llm_explanation"] = explain_match_with_llm(
                    mode="job_to_resumes",
                    similarity_score=h["score"],
                    job_text=job_text,
                    resume_text=resume_text,
                    top_k_rank=h["rank"],
                    client=client,
                )
            except Exception as e:
                explanations_errors.append(f"{h.get('filename')}: {e}")

    return {
        "status": "OK",
        "mode": "job_to_cvs_dataset",
        "top_k": top_k,
        "index_used": {
            "faiss_index_path": faiss_index_path,
            "mapping_path": mapping_path,
        },
        "resume_txt_folder": resume_txt_folder,
        "diagnostics": {
            "faiss_ntotal": faiss_ntotal,
            "mapping_len": mapping_len,
            "resume_txt_folder_exists": resume_txt_folder_exists,
            "add_explanations": add_explanations,
            "explain_top_n": explain_top_n,
            "num_hits": len(hits),
            "explanations_errors": explanations_errors,
        },
        "hits": hits,
    }
