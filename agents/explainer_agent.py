# agents/explainer_agent.py

from __future__ import annotations
from typing import Literal, Optional, Dict
import os
from openai import OpenAI


Mode = Literal["job_to_resumes", "resume_to_jobs", "pair_compatibility"]


def build_llm_client() -> OpenAI:
    # Creating OpenAI client (OpenRouter or OpenAI)
    # If you use OpenRouter, keep base_url="https://openrouter.ai/api/v1"
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def explain_match_with_llm(
    *,
    mode: Mode,
    similarity_score: float,
    job_text: Optional[str] = None,
    resume_text: Optional[str] = None,
    top_k_rank: Optional[int] = None,
    model: str = "openai/gpt-4o-mini",
    client: Optional[OpenAI] = None,
) -> Dict:
    """
    Explaining why a match is relevant using an LLM.

    mode:
      - "job_to_resumes": job is the query, resume is a retrieved candidate
      - "resume_to_jobs": resume is the query, job is a retrieved candidate
      - "pair_compatibility": direct comparison resume <-> job (single pair)

    returns a dict with:
      - explanation: short text
      - strengths: bullet list
      - gaps: bullet list
      - decision: (only for pair_compatibility) "strong_match" | "medium_match" | "weak_match"
    """

    if client is None:
        client = build_llm_client()

    # Basic validation
    if mode in ("job_to_resumes", "pair_compatibility") and not job_text:
        raise ValueError("job_text is required for this mode")
    if mode in ("resume_to_jobs", "pair_compatibility") and not resume_text:
        raise ValueError("resume_text is required for this mode")

    # Building a mode-specific instruction
    if mode == "job_to_resumes":
        header = f"""
You are a recruitment assistant.
We queried the system with a JOB OFFER and retrieved a candidate RESUME.
Your goal is to explain why this resume is relevant to the job.

Context:
- This resume is ranked #{top_k_rank} among the retrieved candidates.
- Similarity score (cosine similarity) = {similarity_score:.3f}
"""
        task = """
Task:
1) Explain in 3-5 lines why this resume matches the job.
2) List 3 key strengths (bullet points).
3) List 2 gaps / missing points (bullet points).
Return ONLY valid JSON with keys: explanation, strengths, gaps.
"""

    elif mode == "resume_to_jobs":
        header = f"""
You are a recruitment assistant.
We queried the system with a RESUME and retrieved a JOB OFFER.
Your goal is to explain why this job offer fits the candidate profile.

Context:
- This job offer is ranked #{top_k_rank} among the retrieved offers.
- Similarity score (cosine similarity) = {similarity_score:.3f}
"""
        task = """
Task:
1) Explain in 3-5 lines why this job offer matches the resume.
2) List 3 key strengths / fit elements (bullet points).
3) List 2 risks / mismatches (bullet points).
Return ONLY valid JSON with keys: explanation, strengths, gaps.
"""

    else:  # pair_compatibility
        header = f"""
You are a recruitment assistant.
We are evaluating the compatibility between ONE resume and ONE job offer.

Context:
- Similarity score (cosine similarity) = {similarity_score:.3f}
"""
        task = """
Task:
1) Provide a short explanation (4-6 lines).
2) List 3 strengths (bullet points).
3) List 3 gaps (bullet points).
4) Give a final decision among: strong_match, medium_match, weak_match.
Return ONLY valid JSON with keys: explanation, strengths, gaps, decision.
"""

    # Injecting the texts
    content = header + "\n" + task + "\n\n"

    if job_text:
        content += f"JOB OFFER:\n{job_text}\n\n"
    if resume_text:
        content += f"RESUME:\n{resume_text}\n\n"

    # Calling LLM
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
    )

    raw = resp.choices[0].message.content.strip()

    # We return raw JSON string in notebook, or parse json safely elsewhere
    # Keeping it simple here
    return {"raw_json": raw}
