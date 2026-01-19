from __future__ import annotations
from typing import Literal, Optional, Dict, Any
from langchain_ollama import ChatOllama # type: ignore
import os
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, Field # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
from agents.prompt_templates import format_messages
from app.matching import safe_json_parse


load_dotenv()

Mode = Literal["job_to_cvs", "cv_to_jobs", "cv_job_fit"]

# ---------------------------
# Output schemas
# ---------------------------
class ExplainPair(BaseModel):
    explanation: str = Field(..., description="4-6 lines explaining compatibility")
    strengths: list[str] = Field(default_factory=list, description="3 bullets")
    gaps: list[str] = Field(default_factory=list, description="3 bullets")
    decision: Literal["strong_match", "medium_match", "weak_match"]
    advice: list[str] = Field(default_factory=list, description="3 actionable tips to improve the resume for THIS job")

class CoachingQA(BaseModel):
    category: Literal["Strength", "Project", "Technical", "Behavioral", "Gap", "Motivation", "Scenario"] = "Strength"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question: str
    suggested_answer: str
    cv_evidence: str = ""
    job_requirement: str = ""
    follow_up: str = ""

class InterviewCoaching(BaseModel):
    questions: list[CoachingQA] = Field(default_factory=list, description="10 tailored interview Q/A pairs")


def _build_llm(backend: str = "OFF"):
    backend = (backend or "OFF").upper()

    if backend == "OFF":
        return None

    if backend in ("OPENROUTER", "OPENAI"):
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env")
        return ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=0.2)

    if backend == "OLLAMA":
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        return ChatOllama(model=model, temperature=0.0)

    raise ValueError(f"Unknown backend: {backend}")

def explain_match_with_llm(
    *,
    mode: Mode,
    similarity_score: float,
    job_text: Optional[str] = None,
    resume_text: Optional[str] = None,
    top_k_rank: Optional[int] = None,
    backend: str = "OLLAMA",  # switch to OPENROUTER or OLLAMA later
) -> Dict[str, Any]:

    if mode in ("job_to_cvs", "cv_job_fit") and not job_text:
        raise ValueError("job_text is required for this mode")
    if mode in ("cv_to_jobs", "cv_job_fit") and not resume_text:
        raise ValueError("resume_text is required for this mode")

    llm = _build_llm(backend)

    if llm is None:
        decision = "medium_match" if similarity_score >= 0.45 else "weak_match"
        parsed = ExplainPair(
            explanation=f"LLM disabled. Similarity score={similarity_score:.3f}.",
            strengths=["Semantic similarity suggests overlap."],
            gaps=["LLM reasoning disabled (backend=OFF)."],
            decision=decision,
            advice=["Enable backend=OPENROUTER or backend=OLLAMA to get real advice."]
        ).model_dump()

        return {"raw_json": "", "parsed": parsed, **parsed}

    schema = ExplainPair

    backend_u = (backend or "OFF").upper()

    if backend_u in ("OPENAI", "OPENROUTER"):
        structured = llm.with_structured_output(schema)

        messages = format_messages(
            mode=mode,
            resume_text=resume_text or "",
            job_text=job_text or "",
            similarity_score=similarity_score,
            top_k_rank=top_k_rank,
        )

        out_obj = structured.invoke(messages)
        parsed = out_obj.model_dump()
        return {"raw_json": "", "parsed": parsed, **parsed}


    messages = format_messages(
        mode=mode,
        resume_text=resume_text or "",
        job_text=job_text or "",
        similarity_score=similarity_score,
        top_k_rank=top_k_rank,
    )

    # Force JSON-only output (messages are dicts)

    if mode == "cv_job_fit":
        messages[-1]["content"] = (
            "IMPORTANT: return JSON with keys: explanation, strengths, gaps, decision, advice. "
            "No other keys.\n"
            + messages[-1]["content"]
        )
    else:
        messages[-1]["content"] = (
            "Return ONLY a valid JSON object. No markdown. No extra text.\n"
            + messages[-1]["content"]
        )

    raw_text = llm.invoke(messages).content or ""
    parsed, err = safe_json_parse(raw_text)


    # Fallback if parse failed
    if err or not isinstance(parsed, dict) or not parsed:
        parsed = {
            "explanation": raw_text.strip()[:1200] if raw_text else "No LLM output.",
            "strengths": [],
            "gaps": [],
        }
        if mode == "cv_job_fit":
            parsed.update({"decision": "weak_match", "advice": []})

    return {"raw_json": raw_text, "parsed": parsed, "parse_error": err, **parsed}

def generate_interview_coaching(
    *,
    similarity_score: float,
    job_text: str,
    resume_text: str,
    backend: str = "OLLAMA",
    n_questions: int = 10,
) -> Dict[str, Any]:
    """
    Generates interview coaching Q/A pairs tailored to the resume + job offer.
    Always returns JSON with: {"questions": [...]}
    """

    llm = _build_llm(backend)
    n_questions = max(3, min(int(n_questions or 10), 20))

    # backend OFF -> deterministic empty coaching
    if llm is None:
        parsed = InterviewCoaching(questions=[]).model_dump()
        return {"raw_json": "", "parsed": parsed, **parsed}

    schema = InterviewCoaching
    backend_u = (backend or "OFF").upper()

    # Messages: keep it simple and strict JSON
    messages = [
        {
            "role": "system",
            "content": (
                "You are an interview coach. "
                "You must return ONLY valid JSON. No markdown. No extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: Build {n_questions} interview questions AND suggested answers tailored to the resume and job offer.\n"
                "Use STAR-style answers when relevant. Include a mix:\n"
                "- Strength/Project questions (value strengths)\n"
                "- Technical questions (match job requirements)\n"
                "- Behavioral questions\n"
                "- Gap/objection handling questions (turn weaknesses into a positive plan)\n"
                "\n"
                "Output schema (JSON only):\n"
                "{\n"
                '  "questions": [\n'
                "    {\n"
                '      "category": "Strength|Project|Technical|Behavioral|Gap|Motivation|Scenario",\n'
                '      "difficulty": "easy|medium|hard",\n'
                '      "question": "...",\n'
                '      "suggested_answer": "...",\n'
                '      "cv_evidence": "short quote or evidence from resume",\n'
                '      "job_requirement": "the requirement this targets",\n'
                '      "follow_up": "one follow-up question"\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "\n"
                f"Similarity score (semantic): {similarity_score:.3f}\n\n"
                "RESUME:\n"
                f"{resume_text}\n\n"
                "JOB OFFER:\n"
                f"{job_text}\n"
            ),
        },
    ]

    # Structured output for OpenRouter/OpenAI path
    if backend_u in ("OPENAI", "OPENROUTER"):
        structured = llm.with_structured_output(schema)
        out_obj = structured.invoke(messages)
        parsed = out_obj.model_dump()
        return {"raw_json": "", "parsed": parsed, **parsed}

    # Ollama path: strict JSON + parse
    raw_text = llm.invoke(messages).content or ""
    parsed, err = safe_json_parse(raw_text)

    if err or not isinstance(parsed, dict) or "questions" not in parsed:
        parsed = {"questions": []}

    # Ensure list type
    if not isinstance(parsed.get("questions"), list):
        parsed["questions"] = []

    return {"raw_json": raw_text, "parsed": parsed, "parse_error": err, **parsed}
