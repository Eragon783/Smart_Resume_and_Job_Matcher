import streamlit as st
import pandas as pd
from app.pipeline import run


def render_hits_table(hits, title="Ranked results"):
    if not hits:
        st.info("No results to display.")
        return

    df = pd.DataFrame(hits)
    for col in ["rank", "filename", "score"]:
        if col not in df.columns:
            df[col] = None

    df["score"] = df["score"].apply(lambda x: round(float(x), 4) if x is not None else None)

    st.markdown(f"### {title}")
    st.dataframe(df[["rank", "filename", "score"]], use_container_width=True, hide_index=True)


def render_hits_cards(hits):
    if not hits:
        return

    st.markdown("### Details")
    for h in hits:
        rank = h.get("rank")
        filename = h.get("filename")
        score = h.get("score")

        header = f"#{rank} — {filename} (score: {score:.4f})" if isinstance(score, (int, float)) else f"#{rank} — {filename}"
        with st.expander(header, expanded=False):
            st.write({
                "rank": rank,
                "filename": filename,
                "score": round(float(score), 6) if isinstance(score, (int, float)) else score,
            })

            if h.get("llm_explanation") is not None:
                st.markdown("**LLM Explanation**")
                st.json(h["llm_explanation"])


def label_from_score(score: float) -> str:
    if score >= 0.45:
        return "Strong fit"
    if score >= 0.30:
        return "Moderate fit"
    return "Weak fit"

import json
import re

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

    llm = out.get("llm")

    # ---- normalize llm into the "final" dict we want to display
    llm_final = None

    if isinstance(llm, dict):
        # Your case: {"raw_json": "```json {...} ```"}
        if "raw_json" in llm and isinstance(llm["raw_json"], str):
            llm_final = _extract_json_from_markdown(llm["raw_json"])
        else:
            # normal dict case
            llm_final = llm

    elif isinstance(llm, str) and llm.strip():
        # sometimes it's directly a string
        llm_final = _extract_json_from_markdown(llm)

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


def main():
    st.set_page_config(page_title="Resume & Job Matcher", layout="wide")
    st.title("Smart Resume & Job Matcher")

    mode_label = st.selectbox(
        "Choose mode",
        [
            "Find the best job offers for your resume",
            "Find the best resumes for a job offer (upload 1–10 resumes)",
            "Find the best resumes in the dataset for a job offer",
            "Evaluate resume–job compatibility",
            "Find the best LinkedIn job offers for your resume",
        ],
    )

    mode = {
        "Find the best job offers for your resume": "cv_to_jobs",
        "Find the best resumes for a job offer (upload 1–10 resumes)": "job_to_cvs_upload",
        "Find the best resumes in the dataset for a job offer": "job_to_cvs_dataset",
        "Evaluate resume–job compatibility": "cv_job_fit",
        "Find the best LinkedIn job offers for your resume": "cv_to_linkedin_jobs",
    }[mode_label]

    st.subheader("Inputs")

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
    # Mode: cv_to_jobs (UI only for now)
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
            add_explanations = st.checkbox("Add LLM explanations + advice (top N)", value=False)
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
    # Mode: job_to_cvs_upload (UI only for now)
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
            add_explanations = st.checkbox("Add LLM explanations + advice (top N)", value=False)
        with col2:
            explain_top_n = st.slider("Explain top N resumes", 1, 10, 3, 1, disabled=(not add_explanations))

    # ---------------------------
    # Mode: job_to_cvs_dataset (FUNCTIONAL)
    # ---------------------------
    elif mode == "job_to_cvs_dataset":
        st.markdown("### Job offer")
        job_offer_file = st.file_uploader("Upload a job offer (TXT)", type=["txt"])

        st.markdown("### Retrieval settings")
        top_k = st.slider("Top K resumes", 1, 20, 10, 1)

        col1, col2 = st.columns([1, 1])
        with col1:
            add_explanations = st.checkbox("Add LLM explanations", value=False)
        with col2:
            explain_top_n = st.slider("Explain top N", 1, 10, 3, 1, disabled=(not add_explanations))

        st.caption("This mode searches the prebuilt resume dataset index (FAISS).")

    # ---------------------------
    # Mode: cv_job_fit (FUNCTIONAL)
    # ---------------------------
    elif mode == "cv_job_fit":
        st.markdown("### Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        st.markdown("### Job offer")
        job_offer_file = st.file_uploader("Upload a job offer (TXT)", type=["txt"])

        add_explanations = st.checkbox("Enable LLM decision + advice", value=True)

    # ---------------------------
    # Mode: cv_to_linkedin_jobs (UI only for now)
    # ---------------------------
    elif mode == "cv_to_linkedin_jobs":
        st.markdown("### Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        st.info("LinkedIn job retrieval will be implemented later.")

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
        if out.get("mode") in {"job_to_cvs_dataset", "cv_to_jobs", "job_to_cvs_upload"}:
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
