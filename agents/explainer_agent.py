from __future__ import annotations

from typing import Literal, Optional, Dict, Any
import os
import json
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
Mode = Literal["job_to_resumes", "resume_to_jobs", "pair_compatibility"]


def build_llm_client() -> OpenAI:
    # ✅ Ensuring .env is loaded even if imported weirdly
    load_dotenv()

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Put it in your .env or export it in your environment."
        )

    return OpenAI(base_url=base_url, api_key=api_key)


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()

    # Handling ```json ... ``` fences
    if text.startswith("```"):
        text = text.strip("`").strip()
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    try:
        return json.loads(text)
    except Exception:
        return None


def explain_match_with_llm(
    *,
    mode: Mode,
    similarity_score: float,
    job_text: Optional[str] = None,
    resume_text: Optional[str] = None,
    top_k_rank: Optional[int] = None,
    model: str = "openai/gpt-4o-mini",
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Returns always:
      - raw_json: str
      - parsed: dict|None

    And if parsed is valid JSON:
      - explanation, strengths, gaps, decision (optional)
    """

    if client is None:
        client = build_llm_client()

    # Basic validation
    if mode in ("job_to_resumes", "pair_compatibility") and not job_text:
        raise ValueError("job_text is required for this mode")
    if mode in ("resume_to_jobs", "pair_compatibility") and not resume_text:
        raise ValueError("resume_text is required for this mode")

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

    content = header + "\n" + task + "\n\n"
    if job_text:
        content += f"JOB OFFER:\n{job_text}\n\n"
    if resume_text:
        content += f"RESUME:\n{resume_text}\n\n"

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
    )

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _try_parse_json(raw)

    out: Dict[str, Any] = {"raw_json": raw, "parsed": parsed}

    if isinstance(parsed, dict):
        for k in ("explanation", "strengths", "gaps", "decision"):
            if k in parsed:
                out[k] = parsed[k]

    return out
