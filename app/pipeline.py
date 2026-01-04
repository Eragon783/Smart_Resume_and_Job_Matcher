# app/pipeline.py
from __future__ import annotations

from typing import Dict, Any
from io import BytesIO
from app.cv_job_fit import handle as cv_job_fit_handler

from pdfminer.high_level import extract_text as pdf_extract_text
from sentence_transformers import SentenceTransformer
import numpy as np

from ingestion.loaders import clean_text  # reuse
from app.matching import search_index  # reuse FAISS search logic + normalization
from agents.explainer_agent import explain_match_with_llm, build_llm_client  # reuse


PIPELINE_VERSION = "v3_reuse_loaders_matching_explainer"


# ---------------------------
# Minimal helpers (only what you don't already have)
# ---------------------------

def _decode_txt_bytes(b: bytes) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")


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


# ---------------------------
# Router
# ---------------------------

def run(mode: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if mode == "job_to_cvs_dataset":
        out = _run_job_to_cvs_dataset(inputs)
        out["pipeline_version"] = PIPELINE_VERSION
        return out

    if mode == "cv_job_fit":
        out = _run_cv_job_fit(inputs)
        out["pipeline_version"] = PIPELINE_VERSION
        return out

    return {
        "status": "NOT_IMPLEMENTED_YET",
        "mode": mode,
        "inputs_received": list(inputs.keys()),
        "pipeline_version": PIPELINE_VERSION,
    }


# ---------------------------
# MODE 1: Dataset retrieval (job offer -> best resumes from FAISS)
# Reusing matching.search_index()
# ---------------------------

def _run_job_to_cvs_dataset(inputs: Dict[str, Any]) -> Dict[str, Any]:
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

    # ✅ reuse matching.py for FAISS loading + normalization + search
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


# ---------------------------
# MODE 2: Pair compatibility (one resume PDF + one job TXT)
# Reusing loaders.clean_text() + explainer_agent.pair_compatibility
# ---------------------------

def _run_cv_job_fit(inputs: Dict[str, Any]) -> Dict[str, Any]:
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
            client = build_llm_client()
            llm = explain_match_with_llm(
                mode="pair_compatibility",
                similarity_score=sim,
                job_text=job_text,
                resume_text=resume_text,
                client=client,
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
