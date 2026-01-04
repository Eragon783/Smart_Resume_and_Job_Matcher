import streamlit as st
import pandas as pd
from app.pipeline import run


def render_hits_table(hits, title="Ranked results"):
    if not hits:
        st.info("No results to display.")
        return

    df = pd.DataFrame(hits)

    # Ensuring expected cols
    for col in ["rank", "filename", "score"]:
        if col not in df.columns:
            df[col] = None

    # Formatting
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

        if isinstance(score, (int, float)):
            header = f"#{rank} — {filename} (score: {score:.4f})"
        else:
            header = f"#{rank} — {filename}"

        with st.expander(header, expanded=False):
            # No CV preview: only metadata + optional explanation
            st.write({
                "rank": rank,
                "filename": filename,
                "score": round(float(score), 6) if isinstance(score, (int, float)) else score,
            })

            if h.get("llm_explanation") is not None:
                st.markdown("**LLM Explanation**")
                st.json(h["llm_explanation"])


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

    resume_file = None
    resume_files = None
    job_offer_file = None
    job_offer_files = None

    n_job_offers = None
    n_resumes = None

    top_k = None
    add_explanations = None
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
    # Mode: cv_job_fit (UI only for now)
    # ---------------------------
    elif mode == "cv_job_fit":
        st.markdown("### Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        st.markdown("### Job offer")
        job_offer_file = st.file_uploader("Upload a job offer (TXT)", type=["txt"])

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

        # ✅ Robust status handling
        status = out.get("status", "UNKNOWN")
        if isinstance(status, str) and status.startswith("OK"):
            st.success("Done ✅")
        elif status == "NOT_IMPLEMENTED_YET":
            st.warning("Mode not implemented yet.")
        else:
            st.error(out.get("error", "An error occurred."))

        # ✅ Always show what keys we got back
        #st.caption(f"Pipeline returned keys: {list(out.keys())}")

        hits = out.get("hits", None)

        if hits is None:
            st.warning("No 'hits' key returned by pipeline. Check pipeline.py return payload.")
        elif not isinstance(hits, list):
            st.warning(f"'hits' is not a list (type={type(hits)}).")
            with st.expander("hits (raw)", expanded=False):
                st.write(hits)
        else:
            #st.caption(f"Number of hits: {len(hits)}")
            render_hits_table(hits)
            render_hits_cards(hits)

        with st.expander("Debug output (raw pipeline response)", expanded=False):
            st.json(out)



if __name__ == "__main__":
    main()
