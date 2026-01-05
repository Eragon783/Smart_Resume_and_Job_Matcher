from __future__ import annotations
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from ingestion.loaders import clean_text
from agents.explainer_agent import explain_match_with_llm
from app.matching import _extract_pdf_text_from_bytes, _cosine_similarity, _decode_txt_bytes

def handle(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mode: cv_to_jobs
    Input:
      - resume_file_bytes (PDF bytes)
      - job_offer_files: list of {"filename": str, "bytes": bytes} (TXT)
      - add_explanations (bool) [optional]
      - explain_top_n (int) [optional]

    Output:
      - hits: ranked job offers with similarity scores (+ optional llm_explanation)
    """
    resume_pdf_bytes = inputs.get("resume_file_bytes")
    job_offer_files = inputs.get("job_offer_files")

    if not resume_pdf_bytes:
        return {"status": "ERROR", "error": "Missing resume_file_bytes (PDF)"}

    if not job_offer_files or not isinstance(job_offer_files, list):
        return {"status": "ERROR", "error": "Missing job_offer_files (upload 1–10 TXT files)"}

    # Hard safety cap
    job_offer_files = job_offer_files[:10]

    # Extracting + cleaning
    resume_text = _extract_pdf_text_from_bytes(resume_pdf_bytes)
    if not resume_text.strip():
        return {"status": "ERROR", "error": "Could not extract resume text from PDF (empty)."}

    job_texts: List[Dict[str, str]] = []
    for f in job_offer_files:
        name = f.get("filename", "unknown.txt")
        b = f.get("bytes", b"")
        txt = clean_text(_decode_txt_bytes(b))
        if txt.strip():
            job_texts.append({"filename": name, "text": txt})

    if not job_texts:
        return {"status": "ERROR", "error": "All uploaded job offers are empty/unreadable."}

    # Embedding model
    model_name = inputs.get("model_name") or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Compute similarity for each job offer (resume is fixed)
    scored = []
    for item in job_texts:
        sim = _cosine_similarity(model, item["text"], resume_text)  # job_text vs resume_text
        scored.append({"filename": item["filename"], "score": float(sim), "job_text": item["text"]})

    # Sort descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Build hits payload
    hits = []
    for rank, s in enumerate(scored, start=1):
        hits.append({
            "rank": rank,
            "filename": s["filename"],
            "score": s["score"],
        })

    # Optional LLM explanations for top N
    add_explanations = bool(inputs.get("add_explanations") or False)
    explain_top_n = int(inputs.get("explain_top_n") or 3)
    explanations_errors = []

    if add_explanations:
        for i in range(min(explain_top_n, len(scored))):
            try:
                # Using existing explainer. We rely on your agent to include advice.
                # If it returns raw_json in markdown, your Streamlit renderer already handles it.
                llm_out = explain_match_with_llm(
                    mode="resume_to_jobs",
                    similarity_score=scored[i]["score"],
                    job_text=scored[i]["job_text"],
                    resume_text=resume_text,
                    top_k_rank=i + 1,
                    backend="OLLAMA",
                )
                hits[i]["llm_explanation"] = llm_out
            except Exception as e:
                explanations_errors.append(f"{scored[i]['filename']}: {e}")

    return {
        "status": "OK",
        "mode": "cv_to_jobs",
        "hits": hits,
        "diagnostics": {
            "model_name": model_name,
            "num_job_offers_used": len(job_texts),
            "llm_enabled": add_explanations,
            "explain_top_n": explain_top_n,
            "explanations_errors": explanations_errors,
        },
    }
