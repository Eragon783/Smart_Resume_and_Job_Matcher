# app/cv_job_fit.py
from __future__ import annotations

from typing import Dict, Any
from io import BytesIO

import numpy as np
from pdfminer.high_level import extract_text as pdf_extract_text
from sentence_transformers import SentenceTransformer

from ingestion.loaders import clean_text  # reuse 
from agents.explainer_agent import explain_match_with_llm, build_llm_client  # reuse 


def _decode_txt_bytes(b: bytes) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")


def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Streamlit gives bytes; loaders.py is path-based.
    Minimal wrapper: bytes -> text -> clean_text().
    """
    if not pdf_bytes:
        return ""
    raw = pdf_extract_text(BytesIO(pdf_bytes))
    return clean_text(raw)


def _cosine_similarity(model: SentenceTransformer, a: str, b: str) -> float:
    """
    Cosine similarity via dot product of normalized vectors.
    """
    va = model.encode([a], convert_to_numpy=True).astype("float32")
    vb = model.encode([b], convert_to_numpy=True).astype("float32")

    va /= (np.linalg.norm(va, axis=1, keepdims=True) + 1e-12)
    vb /= (np.linalg.norm(vb, axis=1, keepdims=True) + 1e-12)

    return float(np.sum(va * vb))


def handle(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mode: cv_job_fit
    Input: resume PDF bytes + job offer TXT bytes
    Output: cosine similarity + optional LLM decision/strengths/gaps/advice
    """
    resume_pdf_bytes = inputs.get("resume_file_bytes")
    job_offer = inputs.get("job_offer_file")

    if not resume_pdf_bytes:
        return {"status": "ERROR", "error": "Missing resume_file_bytes (PDF)"}
    if not job_offer or not job_offer.get("bytes"):
        return {"status": "ERROR", "error": "Missing job_offer_file (TXT)"}

    resume_text = _extract_pdf_text_from_bytes(resume_pdf_bytes)
    job_text = clean_text(_decode_txt_bytes(job_offer["bytes"]))

    if not resume_text.strip():
        return {"status": "ERROR", "error": "Could not extract resume text from PDF (empty)."}
    if not job_text.strip():
        return {"status": "ERROR", "error": "Job offer text is empty/unreadable."}

    model_name = inputs.get("model_name") or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    similarity = _cosine_similarity(model, job_text, resume_text)

    add_explanations = bool(inputs.get("add_explanations") or False)
    llm = None
    llm_error = None

    if add_explanations:
        try:
            client = build_llm_client()
            llm = explain_match_with_llm(
                mode="pair_compatibility",
                similarity_score=similarity,
                job_text=job_text,
                resume_text=resume_text,
                client=client,
            )
        except Exception as e:
            llm_error = str(e)

    return {
        "status": "OK",
        "mode": "cv_job_fit",
        "similarity_score": similarity,
        "llm": llm,  # expected to include decision/strengths/gaps (+ advice if your agent returns it)
        "diagnostics": {
            "model_name": model_name,
            "resume_chars": len(resume_text),
            "job_chars": len(job_text),
            "llm_enabled": add_explanations,
            "llm_error": llm_error,
        },
    }
