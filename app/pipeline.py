from __future__ import annotations
from typing import Dict, Any
from app.graphs.pipeline_graph import build_pipeline_graph
import io
import re
import  fitz # PyMuPDF
from docx import Document


_GRAPH = None

def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_pipeline_graph()
    return _GRAPH

def extract_text_from_file(file_obj: dict) -> str:
    """
    file_obj = {"filename": str, "bytes": bytes}
    """
    filename = file_obj["filename"].lower()
    data = file_obj["bytes"]

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text

    if filename.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    if filename.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {filename}")

def sanitize_text_for_llm(text: str) -> str:
    # Remove null bytes
    text = text.replace("\x00", "")
    # Keep printable ASCII + newlines
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)
    return text.strip()

def smart_trim(text: str, max_chars: int) -> str:
    """Keeping start + end to preserve key info while limiting context."""
    if len(text) <= max_chars:
        return text
    head = text[:2500]
    tail = text[-2000:]
    return head + "\n\n[...TRUNCATED...]\n\n" + tail


def run(mode: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint called by Streamlit:
    out = run(mode, inputs)
    """
    try:
        # --- Implement only cv_job_fit for now ---
        if mode == "cv_job_fit":
            from agents.explainer_agent import explain_match_with_llm
            from sentence_transformers import SentenceTransformer, util  # type: ignore

            resume_file = inputs.get("resume_file")
            job_file = inputs.get("job_offer_file")
            add_explanations = bool(inputs.get("add_explanations", True))

            if not resume_file or not job_file:
                return {"status": "ERROR", "mode": mode, "error": "Missing resume or job offer"}

            # 1) Extract + sanitize
            resume_text = sanitize_text_for_llm(extract_text_from_file(resume_file))
            job_text = sanitize_text_for_llm(extract_text_from_file(job_file))

            # 2) Similarity
            model_name = "all-MiniLM-L6-v2"
            st_model = SentenceTransformer(model_name)
            emb_r = st_model.encode(resume_text, convert_to_tensor=True)
            emb_j = st_model.encode(job_text, convert_to_tensor=True)
            similarity_score = float(util.cos_sim(emb_r, emb_j).item())

            # 3) Optional LLM
            llm_out = None
            llm_error = None
            if add_explanations:
                resume_trim = smart_trim(resume_text, 4500)
                job_trim = smart_trim(job_text, 3000)

                llm_out = explain_match_with_llm(
                    mode="cv_job_fit",
                    similarity_score=similarity_score,
                    resume_text=resume_trim,
                    job_text=job_trim,
                    backend="OLLAMA",
                )
                if isinstance(llm_out, dict):
                    llm_error = llm_out.get("parse_error")

            return {
                "status": "OK",
                "mode": mode,
                "similarity_score": similarity_score,
                "llm": llm_out,
                "diagnostics": {
                    "model_name": model_name,
                    "resume_chars": len(resume_text),
                    "job_chars": len(job_text),
                    "llm_enabled": bool(add_explanations),
                    "llm_error": llm_error,
                },
            }

        # --- Other modes handled by your graph later ---
        return {"status": "NOT_IMPLEMENTED_YET", "mode": mode}

    except Exception as e:
        return {"status": "ERROR", "mode": mode, "error": repr(e)}

