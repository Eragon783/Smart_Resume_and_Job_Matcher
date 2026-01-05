
def format_messages(
    *,
    mode: str,
    resume_text: str = "",
    job_text: str = "",
    similarity_score: float | None = None,
    top_k_rank: int | None = None,
):
    """
    Build chat-style messages for Ollama / LLM.

    Returns:
        List[dict]: [{"role": "system"|"user", "content": "..."}]
    """

    system_prompt = (
        "You are an expert recruitment assistant.\n"
        "Your task is to explain resume/job compatibility clearly and concisely.\n"
        "Only use information present in the provided texts.\n"
        "Do not invent skills or experiences.\n"
        "Use bullet points when appropriate."
    )

    # -------------------------
    # MODE: Resume ↔ Job Fit
    # -------------------------
    if mode == "cv_job_fit":
        user_prompt = f"""
Analyze the compatibility between the following resume and job offer.

Cosine similarity score: {similarity_score}

JOB OFFER:
{job_text}

RESUME:
{resume_text}

Return:
1. A short explanation (3–6 lines)
2. Strengths (bullet points)
3. Gaps / missing elements (bullet points)
4. Concrete suggestions to improve the resume for this job (bullet points)
"""

    # -------------------------
    # MODE: One Job → Many Resumes
    # -------------------------
    elif mode == "job_to_cvs":
        rank_info = f"Rank: #{top_k_rank}" if top_k_rank else ""

        user_prompt = f"""
Explain why the following resume matches the job offer.

{rank_info}
Cosine similarity score: {similarity_score}

JOB OFFER:
{job_text}

RESUME:
{resume_text}

Return:
- A short explanation (3–5 lines)
- Key matching points (bullet points)
"""

    # -------------------------
    # MODE: One Resume → Many Jobs
    # -------------------------
    elif mode == "cv_to_jobs":
        rank_info = f"Rank: #{top_k_rank}" if top_k_rank else ""

        user_prompt = f"""
Explain why the following job offer matches the resume.

{rank_info}
Cosine similarity score: {similarity_score}

RESUME:
{resume_text}

JOB OFFER:
{job_text}

Return:
- A short explanation (3–5 lines)
- Key matching points (bullet points)
"""

    # -------------------------
    # Fallback (safety)
    # -------------------------
    else:
        user_prompt = f"""
Explain the relationship between the following texts.

TEXT A:
{resume_text}

TEXT B:
{job_text}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()},
    ]