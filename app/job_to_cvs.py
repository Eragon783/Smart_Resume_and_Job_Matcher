from __future__ import annotations
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from ingestion.loaders import clean_text
from agents.explainer_agent import explain_match_with_llm, build_llm_client
from app.cv_job_fit import _extract_pdf_text_from_bytes, _cosine_similarity
from app.matching import _decode_txt_bytes

def handle(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mode: job_to_cvs_upload

    Inputs:
      - job_offer_file: {"filename": str, "bytes": bytes} (TXT)
      - resume_files: list of {"filename": str, "bytes": bytes} (PDF)
      - add_explanations: bool [optional]
      - explain_top_n: int [optional]

    Output:
      - hits: ranked resumes (rank/filename/score) + optional llm_explanation for top N
    """
    job_offer = inputs.get("job_offer_file")
    resume_files = inputs.get("resume_files")

    if not job_offer or not job_offer.get("bytes"):
        return {"status": "ERROR", "error": "Missing job_offer_file (TXT)"}

    if not resume_files or not isinstance(resume_files, list):
        return {"status": "ERROR", "error": "Missing resume_files (upload 1–10 PDFs)"}

    # Safety cap
    resume_files = resume_files[:10]

    job_text = clean_text(_decode_txt_bytes(job_offer["bytes"]))
    if not job_text.strip():
        return {"status": "ERROR", "error": "Job offer text is empty/unreadable."}

    # Extract resume texts
    resumes: List[Dict[str, str]] = []
    for f in resume_files:
        name = f.get("filename", "unknown.pdf")
        b = f.get("bytes", b"")
        resume_text = _extract_pdf_text_from_bytes(b)
        resume_text = clean_text(resume_text)

        if resume_text.strip():
            resumes.append({"filename": name, "text": resume_text})

    if not resumes:
        return {"status": "ERROR", "error": "All uploaded resumes are empty/unreadable after PDF extraction."}

    model_name = inputs.get("model_name") or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    scored = []
    for r in resumes:
        sim = _cosine_similarity(model, job_text, r["text"])
        scored.append({"filename": r["filename"], "score": float(sim), "resume_text": r["text"]})

    scored.sort(key=lambda x: x["score"], reverse=True)

    hits = []
    for rank, s in enumerate(scored, start=1):
        hits.append({"rank": rank, "filename": s["filename"], "score": s["score"]})

    # Optional LLM explanation + advice for top N
    add_explanations = bool(inputs.get("add_explanations") or False)
    explain_top_n = int(inputs.get("explain_top_n") or 3)
    explanations_errors = []

    if add_explanations:
        client = build_llm_client()
        for i in range(min(explain_top_n, len(scored))):
            try:
                llm_out = explain_match_with_llm(
                    mode="job_to_resumes",
                    similarity_score=scored[i]["score"],
                    job_text=job_text,
                    resume_text=scored[i]["resume_text"],
                    top_k_rank=i + 1,
                    client=client,
                )
                hits[i]["llm_explanation"] = llm_out
            except Exception as e:
                explanations_errors.append(f"{scored[i]['filename']}: {e}")

    return {
        "status": "OK",
        "mode": "job_to_cvs_upload",
        "hits": hits,
        "diagnostics": {
            "model_name": model_name,
            "num_resumes_used": len(resumes),
            "llm_enabled": add_explanations,
            "explain_top_n": explain_top_n,
            "explanations_errors": explanations_errors,
        },
    }
