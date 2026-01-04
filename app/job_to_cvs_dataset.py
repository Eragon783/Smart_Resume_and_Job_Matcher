from typing import Dict, Any
import os
from sentence_transformers import SentenceTransformer

from app.pipeline._shared import (
    decode_txt_bytes,
    encode_normalized,
    load_faiss_index,
    faiss_search,
    load_resume_text,
)
from ingestion.loaders import clean_text
from agents.explainer_agent import explain_match_with_llm, build_llm_client


def handle(inputs: Dict[str, Any]) -> Dict[str, Any]:
    job_offer = inputs.get("job_offer_file")
    if not job_offer or not job_offer.get("bytes"):
        return {"status": "ERROR", "error": "Missing job offer TXT"}

    job_text = clean_text(decode_txt_bytes(job_offer["bytes"]))
    if not job_text.strip():
        return {"status": "ERROR", "error": "Empty job offer text"}

    faiss_path = inputs.get("resume_faiss_path", "./data/resume_index.faiss")
    mapping_path = inputs.get("resume_mapping_path", "./data/resume_index_mapping.json")
    resume_txt_folder = inputs.get("resume_txt_folder", "./data/resume_extract_text")

    if not os.path.exists(faiss_path) or not os.path.exists(mapping_path):
        return {"status": "ERROR", "error": "Resume FAISS index not found"}

    index, mapping = load_faiss_index(faiss_path, mapping_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = encode_normalized(model, job_text)

    top_k = int(inputs.get("top_k", 10))
    hits = faiss_search(index, mapping, query_vec, top_k)

    if inputs.get("add_explanations"):
        client = build_llm_client()
        for h in hits[:3]:
            resume_text = load_resume_text(resume_txt_folder, h["filename"])
            h["llm_explanation"] = explain_match_with_llm(
                mode="job_to_resumes",
                similarity_score=h["score"],
                job_text=job_text,
                resume_text=resume_text,
                top_k_rank=h["rank"],
                client=client,
            )

    return {
        "status": "OK",
        "mode": "job_to_cvs_dataset",
        "hits": hits,
    }
