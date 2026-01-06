def format_messages(
    *,
    mode: str,
    resume_text: str = "",
    job_text: str = "",
    similarity_score: float | None = None,
    top_k_rank: int | None = None,
):
    system_prompt = (
        "You are a senior technical recruiter and career coach.\n"
        "Your explanations are precise, evidence-based, and non-generic.\n"
        "You MUST return ONLY valid raw JSON (no markdown, no code fences).\n"
        "Use ONLY the information available in the provided fields.\n"
        "Do NOT invent companies, degrees, years, job titles, or technologies not present.\n"
        "Avoid generic claims (e.g., 'team player') unless explicitly supported.\n"
        "When possible, include short evidence quotes (3–10 words) from the texts.\n"
        "Always map strengths/gaps to job requirements.\n"
    )

    rank_info = f"Rank: #{top_k_rank}" if top_k_rank else "Rank: N/A"
    sim_info = f"{similarity_score:.3f}" if isinstance(similarity_score, (int, float)) else "N/A"

    resume_block = resume_text.strip() if resume_text and resume_text.strip() else "N/A"
    job_block = job_text.strip() if job_text and job_text.strip() else "N/A"

    user_prompt = f"""
Mode: {mode}
{rank_info}
Cosine similarity score: {sim_info}

MODE-SPECIFIC GOAL:
- If mode == "cv_to_jobs": explain why THIS job offer matches THIS resume (use resume evidence).
- If mode == "job_to_cvs": explain why THIS resume matches THIS job (map to job requirements).
- If mode == "cv_job_fit": be the most detailed and strict.

JOB CONTEXT:
{job_block}

RESUME CONTEXT:
{resume_block}

Return ONLY raw JSON with EXACTLY these keys (no extra keys):
{{
  "explanation": "string (4-6 lines, must mention if context is partial)",
  "strengths": ["string", "string", "string"],
  "gaps": ["string", "string", "string"],
  "decision": "strong_match|medium_match|weak_match",
  "advice": ["string", "string", "string"]
}}

HARD RULES (quality requirements):
- No generic bullets. 
- Each gap must reference a missing/unclear requirement from the JOB text.
- Advice must be actionable for THIS exact pair (what to add/rewrite, keywords, proof projects).
- You may use the similarity score only as a weak signal when context is limited.
- Never output extra keys (TEXT_A, TEXT_B, job_title, required_skills, ideal_candidate, compatibility, reasons, etc.).
- No markdown, no code block, ONLY JSON.
- No generic bullets. Every bullet must be specific to THIS pair.
- Avoid generic claims (team player, communication, motivated, fast learner) unless explicitly supported in the text.
- In "explanation", cover in order: (1) top 2 requirements, (2) best evidence, (3) biggest gap, (4) justify decision.
- If context is partial/"N/A", mention low confidence and do NOT output strong_match.

"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()},
    ]
