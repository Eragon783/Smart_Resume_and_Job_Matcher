def format_messages(
    *,
    mode: str,
    resume_text: str = "",
    job_text: str = "",
    similarity_score: float | None = None,
    top_k_rank: int | None = None,
):
    system_prompt = (
        "You are an expert recruitment assistant.\n"
        "You MUST return ONLY valid raw JSON (no markdown, no code fences).\n"
        "Use ONLY the information available in the provided fields.\n"
        "If some information is missing, say so explicitly and give best-effort advice.\n"
        "Do not invent companies, degrees, years, or specific technologies not present.\n"
    )

    rank_info = f"Rank: #{top_k_rank}" if top_k_rank else "Rank: N/A"
    sim_info = f"{similarity_score:.3f}" if isinstance(similarity_score, (int, float)) else "N/A"

    # Important: allow empty texts (dataset modes)
    resume_block = resume_text.strip() if resume_text and resume_text.strip() else "N/A"
    job_block = job_text.strip() if job_text and job_text.strip() else "N/A"

    user_prompt = f"""
Mode: {mode}
{rank_info}
Cosine similarity score: {sim_info}

JOB CONTEXT (may be partial, may be only a title/url/filename):
{job_block}

RESUME CONTEXT (may be partial, may be only a filename/id):
{resume_block}

Return ONLY raw JSON with EXACTLY these keys (no extra keys):
{{
  "explanation": "string (4-6 lines, mention if context is partial)",
  "strengths": ["string", "string", "string"],
  "gaps": ["string", "string", "string"],
  "decision": "strong_match|medium_match|weak_match",
  "advice": ["string", "string", "string"]
}}

Rules:
- If JOB or RESUME context is partial/missing, your explanation MUST say it clearly.
- Strengths/gaps/advice MUST be best-effort and grounded in what is present.
- You may use the similarity score to decide strong/medium/weak when context is limited.
- Never output keys like TEXT_A, TEXT_B, job_title, required_skills, ideal_candidate, compatibility, reasons.
- No markdown, no code block, ONLY JSON.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()},
    ]
