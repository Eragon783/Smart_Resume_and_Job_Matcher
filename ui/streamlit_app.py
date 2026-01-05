import streamlit as st
import pandas as pd
from app.pipeline import run
import json
import re
import urllib.request
#######################################################
import os, sys, platform
st.caption(f"Python: {sys.executable}")
st.caption(f"Platform: {platform.platform()}")
st.caption(f"CWD: {os.getcwd()}")

import requests

def ollama_generate_test(model="llama3.1:8b"):
    payload = {
        "model": model,
        "prompt": "Reply with exactly: OK_OLLAMA",
        "stream": False
    }
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)
    r.raise_for_status()
    return r.json().get("response", "")

if st.button("Test Ollama generation"):
    try:
        txt = ollama_generate_test()
        st.write(txt)
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
def normalize_llm_payload(llm_any):
    """
    Accepts various LLM formats and returns a dict or None.
    Supported:
      - dict already parsed
      - dict with {"raw_json": "```json {...}```"}
      - string containing ```json {...}```
      - string that is JSON
    """
    if llm_any is None:
        return None

    if isinstance(llm_any, dict):
        if "raw_json" in llm_any and isinstance(llm_any["raw_json"], str):
            parsed = _extract_json_from_markdown(llm_any["raw_json"])
            return parsed if isinstance(parsed, dict) else None
        return llm_any

    if isinstance(llm_any, str) and llm_any.strip():
        parsed = _extract_json_from_markdown(llm_any)
        return parsed if isinstance(parsed, dict) else None

    return None


def render_hits_table(hits):
    if not hits:
        st.info("No results.")
        return

    rows = []
    for h in hits:
        rank = h.get("rank")
        score = h.get("score")
        url = h.get("url")
        title = h.get("title")
        filename = h.get("filename")
        index_id = h.get("index_id")

        label = title or filename or url or (f"item_{index_id}" if index_id is not None else "item")

        row = {
            "rank": rank,
            "label": label,
            "score": score,
        }

        # include url only if present (LinkedIn mode)
        if url:
            row["url"] = url

        # keep old compatibility info if present
        if filename and not title:
            row["filename"] = filename

        rows.append(row)

    df = pd.DataFrame(rows)

    # Choosing a stable column order based on availability
    cols = ["rank", "label", "score"]
    if "url" in df.columns:
        cols.insert(2, "url")  # rank, label, url, score
    if "filename" in df.columns and "url" not in df.columns:
        cols.insert(2, "filename")

    # Keeping any extra columns (rare) at the end
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]

    st.dataframe(df[cols], use_container_width=True)


def render_hits_cards(hits):

    if not hits:
        st.info("No results.")
        return

    for h in hits:
        rank = h.get("rank")
        score = h.get("score")
        title = h.get("title")
        filename = h.get("filename")
        url = h.get("url")
        index_id = h.get("index_id")

        label = title or filename or url or (f"item_{index_id}" if index_id is not None else "item")

        with st.container(border=True):
            if isinstance(score, (int, float)):
                st.markdown(f"**N°{rank} — {label}**  \nScore: `{score:.4f}`")
            else:
                st.markdown(f"**N°{rank} — {label}**")

            # LinkedIn: clickable link if present
            if url:
                st.markdown(f"[Open on LinkedIn]({url})")

            if filename and label != filename:
                st.caption(f"File: {filename}")

            llm_final = normalize_llm_payload(h.get("llm_explanation"))

            if llm_final:
                with st.expander("LLM decision"):
                    render_llm_explanation_pretty(llm_final)
            elif h.get("llm_explanation") is not None:
                # If something exists but isn't parseable -> show raw for debugging
                with st.expander("LLM raw output (unparsed)"):
                    st.write(h.get("llm_explanation"))


            # Debug always available
            with st.expander("Debug (raw hit)"):
                st.json(h)

def label_from_score(score: float) -> str:
    if score >= 0.60:
        return "Strong fit"
    if score >= 0.30:
        return "Moderate fit"
    return "Weak fit"

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return [str(x)]

def _extract_json_from_markdown(s: str) -> dict | None:
    """
    Extracting JSON that may be wrapped like:
    ```json
    {...}
    ```
    """
    if not s or not isinstance(s, str):
        return None

    # Try to find a ```json ... ``` block first
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: try to parse the whole string as JSON
    s2 = s.strip()
    try:
        return json.loads(s2)
    except Exception:
        return None

def render_cv_job_fit(out: dict):
    st.markdown("## Compatibility result")

    score = float(out.get("similarity_score", 0.0))
    st.metric("Cosine similarity", f"{score:.4f}", label_from_score(score))

    llm = out.get("llm_explanation") or out.get("llm")
    llm_final = normalize_llm_payload(llm)

    if not isinstance(llm_final, dict):
        st.info("LLM output is present but could not be parsed into JSON.")
        with st.expander("LLM raw output", expanded=True):
            st.write(llm)
        with st.expander("Diagnostics", expanded=False):
            st.json(out.get("diagnostics", {}))
        return

    decision = llm_final.get("decision", "N/A")
    explanation = llm_final.get("explanation", "")

    strengths = llm_final.get("strengths", [])
    gaps = llm_final.get("gaps", [])
    advice = llm_final.get("advice") or llm_final.get("recommendations") or llm_final.get("improvements")

    st.markdown("### LLM decision")
    st.write("**Decision:**", decision)
    if explanation:
        st.write("**Explanation:**")
        st.write(explanation)

    st.markdown("### Strengths")
    s_list = _as_list(strengths)
    if s_list:
        for s in s_list:
            st.write(f"- {s}")
    else:
        st.write("- (none)")

    st.markdown("### Gaps")
    g_list = _as_list(gaps)
    if g_list:
        for g in g_list:
            st.write(f"- {g}")
    else:
        st.write("- (none)")

    st.markdown("### Advice to improve your CV for this job")
    a_list = _as_list(advice) if advice is not None else []
    if a_list:
        for a in a_list:
            st.write(f"- {a}")
    else:
        # fallback: derive from gaps
        if g_list:
            for g in g_list:
                st.write(f"- Add or strengthen: {g}")
        else:
            st.write("- Add more job-specific keywords and measurable achievements.")

    with st.expander("LLM raw output (parsed)", expanded=False):
        st.json(llm_final)

    with st.expander("Diagnostics", expanded=False):
        st.json(out.get("diagnostics", {}))
        

def render_llm_explanation_pretty(llm: dict):
    """
    Pretty display for LLM output across ALL modes.
    Renders: Decision, Explanation, Strengths, Gaps
    Does NOT render Advice (by design).
    """
    if not isinstance(llm, dict):
        st.warning("LLM explanation is not a dict.")
        st.write(llm)
        return

    # Flexible key support (depending on your agent outputs)
    decision = llm.get("decision") or llm.get("match_decision") or llm.get("label")
    explanation = llm.get("explanation") or llm.get("reason") or llm.get("summary")

    strengths = llm.get("strengths") or llm.get("pros") or llm.get("positive_points")
    gaps = llm.get("gaps") or llm.get("missing") or llm.get("cons") or llm.get("negative_points")

    st.markdown("## LLM decision")

    if decision:
        st.markdown(f"**Decision:** `{decision}`")

    if explanation:
        st.markdown("**Explanation:**")
        st.write(explanation)

    if strengths:
        st.markdown("## Strengths")
        for s in _as_list(strengths):
            st.markdown(f"- {s}")

    if gaps:
        st.markdown("## Gaps")
        for g in _as_list(gaps):
            st.markdown(f"- {g}")

    with st.expander("LLM raw output (parsed)"):
        st.json(llm)

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
        "Find the best resumes for a job offer": "job_to_cvs_upload",
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
    elif mode == "job_to_cvs_upload":
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
        if out.get("mode") in {"job_to_cvs_dataset", "cv_to_jobs", "job_to_cvs_upload", "cv_to_linkedin_jobs_dataset"}:
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
