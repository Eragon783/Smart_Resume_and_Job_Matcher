from __future__ import annotations
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from ingestion.loaders import clean_text  # reuse 
from agents.explainer_agent import explain_match_with_llm, build_llm_client  # reuse 
from app.matching import _decode_txt_bytes, _extract_pdf_text_from_bytes, _cosine_similarity

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

    sim = _cosine_similarity(model, job_text, resume_text)

    add_explanations = bool(inputs.get("add_explanations") or False)
    llm = None
    llm_error = None

    if add_explanations:
        try:
            llm = explain_match_with_llm(
                mode="pair_compatibility",
                similarity_score=sim,
                job_text=job_text,
                resume_text=resume_text,
                backend="OPENROUTER",
            )
        except Exception as e:
            llm_error = str(e)

    return {
        "status": "OK",
        "mode": "cv_job_fit",
        "similarity_score": sim,
        "llm": llm,
        "diagnostics": {
            "model_name": model_name,
            "resume_chars": len(resume_text),
            "job_chars": len(job_text),
            "llm_enabled": add_explanations,
            "llm_error": llm_error,
        },
    }