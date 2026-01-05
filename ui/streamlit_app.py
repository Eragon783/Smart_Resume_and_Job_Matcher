import streamlit as st # type: ignore
import pandas as pd # type: ignore
from app.pipeline import run
import json
import re
import urllib.request
from agents.structuring_agent import render_hits_table, render_hits_cards, render_cv_job_fit

#######################################################
import os, sys, platform
st.caption(f"Python: {sys.executable}")
st.caption(f"Platform: {platform.platform()}")
st.caption(f"CWD: {os.getcwd()}")

import requests # type: ignore

def ollama_generate_test(model="llama3.1:latest"):
    url = "http://localhost:11434/api/generate" 
    payload = {"model": model, "prompt": "Reply with exactly: OK_OLLAMA", "stream": False}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("response", "")

if st.button("Test Ollama generation"):
    try:
        st.success(ollama_generate_test()) 
    except Exception as e:
        st.error(str(e))



def ollama_healthcheck(base_url: str = "http://localhost:11434", timeout: int = 2):
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        models = [m.get("name") for m in data.get("models", [])]
        return True, models, None
    except Exception as e:
        return False, [], str(e)

ok, models, err = ollama_healthcheck()
if ok:
    st.success(f"Ollama OK @ http://localhost:11434 — {len(models)} models")
    with st.expander("Models"):
        st.write(models)
else:
    st.error(f"Ollama NOT reachable: {err}")

################################################################################

def main():
    st.set_page_config(page_title="Resume & Job Matcher", layout="wide")
    st.title("Smart Resume & Job Matcher")

    mode_label = st.selectbox(
        "Choose mode",
        [
            "Find the best job offers for your resume",
            "Find the best resumes for a job offer",
            "Find the best resumes in the dataset for a job offer",
            "Find the best LinkedIn's dataset job offers for your resume",
            "Evaluate resume–job compatibility",
        ],
    )

    mode = {
        "Find the best job offers for your resume": "cv_to_jobs",
        "Find the best resumes for a job offer": "job_to_cvs",
        "Find the best resumes in the dataset for a job offer": "job_to_cvs_dataset",
        "Find the best LinkedIn's dataset job offers for your resume": "cv_to_linkedin_jobs_dataset",
        "Evaluate resume–job compatibility": "cv_job_fit",
    }[mode_label]

    # Defaults so inputs dict always has valid keys
    resume_file = None
    resume_files = None
    job_offer_file = None
    job_offer_files = None

    n_job_offers = None
    n_resumes = None

    top_k = None
    add_explanations = False
    explain_top_n = None

    # ---------------------------
    # Mode: cv_to_jobs upload one resume and several job offers
    # ---------------------------
    if mode == "cv_to_jobs":
        st.markdown("### Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        st.markdown("### Job offers")
        n_job_offers = st.slider("How many job offers?", 1, 10, 3, 1)
        job_offer_files = st.file_uploader(
            "Upload job offer TXT files",
            type=["txt"],
            accept_multiple_files=True,
        )
        
        st.markdown("### Ranking settings")
        col1, col2 = st.columns([1, 1])
        with col1:
            add_explanations = st.checkbox("Add LLM explanations", value=True)
        with col2:
            explain_top_n = st.slider("Explain top N job offers", 1, 10, 3, 1, disabled=(not add_explanations))


        if job_offer_files:
            if len(job_offer_files) > n_job_offers:
                st.warning(f"Only the first {n_job_offers} job offer(s) will be used.")
                job_offer_files = job_offer_files[:n_job_offers]
            elif len(job_offer_files) < n_job_offers:
                st.warning(f"You selected {n_job_offers}, but uploaded {len(job_offer_files)}.")
            st.caption(f"Using {len(job_offer_files)} job offer(s).")

    # ---------------------------
    # Mode: job_to_cvs_upload uplaod one job offer and several resumes
    # ---------------------------
    elif mode == "job_to_cvs":
        st.markdown("### Job offer")
        job_offer_file = st.file_uploader("Upload a job offer (TXT)", type=["txt"])

        st.markdown("### Resumes")
        n_resumes = st.slider("How many resumes?", 1, 10, 3, 1)
        resume_files = st.file_uploader(
            "Upload resume PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if resume_files:
            if len(resume_files) > n_resumes:
                st.warning(f"Only the first {n_resumes} resume(s) will be used.")
                resume_files = resume_files[:n_resumes]
            elif len(resume_files) < n_resumes:
                st.warning(f"You selected {n_resumes}, but uploaded {len(resume_files)}.")
            st.caption(f"Using {len(resume_files)} resume(s).")

        st.markdown("### Ranking settings")
        col1, col2 = st.columns([1, 1])
        with col1:
            add_explanations = st.checkbox("Add LLM explanations", value=True)
        with col2:
            explain_top_n = st.slider("Explain top N resumes", 1, 10, 3, 1, disabled=(not add_explanations))

    # ---------------------------
    # Mode: job_to_cvs_dataset
    # ---------------------------
    elif mode == "job_to_cvs_dataset":
        st.markdown("### Job offer")
        job_offer_file = st.file_uploader("Upload a job offer (TXT)", type=["txt"])

        st.markdown("### Retrieval settings")
        top_k = st.slider("Top K resumes", 1, 20, 10, 1)

        col1, col2 = st.columns([1, 1])
        with col1:
            add_explanations = st.checkbox("Add LLM explanations", value=True)
        with col2:
            explain_top_n = st.slider("Explain top N", 1, 10, 3, 1, disabled=(not add_explanations))

        st.caption("This mode searches the prebuilt resume dataset index (FAISS).")

    # ---------------------------
    # Mode: cv_to_linkedin_jobs 
    # ---------------------------
    elif mode == "cv_to_linkedin_jobs_dataset":
        st.markdown("### Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        st.markdown("### Ranking settings")
        top_k = st.slider("Top K LinkedIn job offers", 1, 20, 10, 1)

        st.caption("This mode searches the prebuilt LinkedIn jobs FAISS index (data/job_treated).")

    # ---------------------------
    # Mode: cv_job_fit
    # ---------------------------
    elif mode == "cv_job_fit":
        st.markdown("### Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        st.markdown("### Job offer")
        job_offer_file = st.file_uploader("Upload a job offer (TXT)", type=["txt"])

        add_explanations = st.checkbox("Enable LLM decision + advice", value=True)


    # ---------------------------
    # RUN
    # ---------------------------
    if st.button("Run"):
        inputs = {
            "resume_file_bytes": resume_file.getvalue() if resume_file else None,
            "job_offer_file": (
                {"filename": job_offer_file.name, "bytes": job_offer_file.getvalue()}
                if job_offer_file else None
            ),
            "job_offer_files": (
                [{"filename": f.name, "bytes": f.getvalue()} for f in job_offer_files]
                if job_offer_files else None
            ),
            "resume_files": (
                [{"filename": f.name, "bytes": f.getvalue()} for f in resume_files]
                if resume_files else None
            ),
            "top_k": top_k,
            "add_explanations": add_explanations,
            "explain_top_n": explain_top_n,
        }

        out = run(mode, inputs)

        status = out.get("status", "UNKNOWN")
        if isinstance(status, str) and status.startswith("OK"):
            st.success("Done ✅")
        elif status == "NOT_IMPLEMENTED_YET":
            st.warning("Mode not implemented yet.")
        else:
            st.error(out.get("error", "An error occurred."))

        # ---------------------------
        # Display by mode
        # ---------------------------
        if out.get("mode") in {"job_to_cvs_dataset", "cv_to_jobs", "job_to_cvs", "cv_to_linkedin_jobs_dataset"}:
            hits = out.get("hits")
            if isinstance(hits, list):
                st.caption(f"Pipeline returned {len(hits)} result(s).")
                render_hits_table(hits)
                render_hits_cards(hits)

        elif out.get("mode") == "cv_job_fit" and out.get("status", "").startswith("OK"):
            render_cv_job_fit(out)

        with st.expander("Debug output (raw pipeline response)", expanded=False):
            st.json(out)


if __name__ == "__main__":
    main()
