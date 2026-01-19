from __future__ import annotations
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from ingestion.loaders import clean_text  # reuse 
from agents.explainer_agent import explain_match_with_llm
from sentence_transformers import SentenceTransformer, util #type: ignorereuse 
from utils.json_utils import extract_text_from_file, sanitize_text_for_llm, smart_trim

def handle(inputs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resume_file = inputs.get("resume_file")
        job_file = inputs.get("job_offer_file")
        add_explanations = bool(inputs.get("add_explanations", True))

        if not resume_file or not job_file:
            return {"status": "ERROR", "mode": "cv_job_fit", "error": "Missing resume or job offer"}

        # 1) Extract + sanitize
        resume_text = sanitize_text_for_llm(extract_text_from_file(resume_file))
        job_text = sanitize_text_for_llm(extract_text_from_file(job_file))

        # 2) Similarity
        model_name = "all-MiniLM-L6-v2"
        st_model = SentenceTransformer(model_name)
        emb_r = st_model.encode(resume_text, convert_to_tensor=True)
        emb_j = st_model.encode(job_text, convert_to_tensor=True)
        similarity_score = float(util.cos_sim(emb_r, emb_j).item())

        backend = (inputs.get("llm_backend") or "OLLAMA").upper()

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
                backend=backend,
            )
            if isinstance(llm_out, dict):
                llm_error = llm_out.get("parse_error")
        print("LLM backend used:", backend)
        return {
            "status": "OK",
            "mode": "cv_job_fit",
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

    except Exception as e:
        return {"status": "ERROR", "mode": "cv_job_fit", "error": repr(e)}

