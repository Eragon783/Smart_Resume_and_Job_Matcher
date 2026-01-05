from typing import Dict, Any
import os
from ingestion.loaders import clean_text
from agents.explainer_agent import explain_match_with_llm
from app.matching import search_index, _decode_txt_bytes

def handle(inputs: Dict[str, Any]) -> Dict[str, Any]:
    job_offer = inputs.get("job_offer_file")
    if not job_offer or not job_offer.get("bytes"):
        return {"status": "ERROR", "error": "Missing job_offer_file (TXT)"}

    job_text = clean_text(_decode_txt_bytes(job_offer["bytes"]))
    if not job_text.strip():
        return {"status": "ERROR", "error": "Job offer text is empty/unreadable"}

    index_path = inputs.get("resume_faiss_path") or "./data/resume_index.faiss"
    mapping_path = inputs.get("resume_mapping_path") or "./data/resume_index_mapping.json"
    top_k = int(inputs.get("top_k") or 10)
    model_name = inputs.get("model_name") or "all-MiniLM-L6-v2"

    hits = search_index(
        query_text=job_text,
        index_path=index_path,
        mapping_path=mapping_path,
        top_k=top_k,
        model_name=model_name,
    )

    # Optional explanations (top N only)
    add_explanations = bool(inputs.get("add_explanations") or False)
    explain_top_n = int(inputs.get("explain_top_n") or 3)

    # IMPORTANT: explanations require resume text files; if you don't want to preview,
    # we still need to load the text internally for the LLM.
    resume_txt_folder = inputs.get("resume_txt_folder") or "./data/resume_extract_text"

    explanations_errors = []
    if add_explanations and hits:
        client = build_llm_client()
        for h in hits[:min(explain_top_n, len(hits))]:
            filename = h.get("filename")
            try:
                # Internal load only, no UI preview
                path = f"{resume_txt_folder}/{filename}"
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    resume_text = clean_text(f.read())

                h["llm_explanation"] = explain_match_with_llm(
                    mode="job_to_resumes",
                    similarity_score=h["score"],
                    job_text=job_text,
                    resume_text=resume_text,
                    top_k_rank=h["rank"],
                    client=client,
                )
            except Exception as e:
                explanations_errors.append(f"{filename}: {e}")

    return {
        "status": "OK",
        "mode": "job_to_cvs_dataset",
        "hits": hits,
        "diagnostics": {
            "index_path": index_path,
            "mapping_path": mapping_path,
            "resume_txt_folder": resume_txt_folder,
            "model_name": model_name,
            "add_explanations": add_explanations,
            "explain_top_n": explain_top_n,
            "num_hits": len(hits),
            "explanations_errors": explanations_errors,
        },
    }
