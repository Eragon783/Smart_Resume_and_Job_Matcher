from __future__ import annotations
from typing import Literal, Optional, Dict, Any
from langchain_ollama import ChatOllama # type: ignore
import os
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, Field # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
import json
import re
import json5 # type: ignore
from agents.prompt_templates import format_messages
from utils.json_utils import ollama_chat
from utils.json_utils import safe_json_parse


load_dotenv()

Mode = Literal["job_to_cvs", "cv_to_jobs", "cv_job_fit"]

# ---------------------------
# Output schemas
# ---------------------------
class ExplainRanked(BaseModel):
    explanation: str = Field(..., description="3-5 lines explaining the match")
    strengths: list[str] = Field(default_factory=list, description="3 bullets")
    gaps: list[str] = Field(default_factory=list, description="2 bullets")

class ExplainPair(BaseModel):
    explanation: str = Field(..., description="4-6 lines explaining compatibility")
    strengths: list[str] = Field(default_factory=list, description="3 bullets")
    gaps: list[str] = Field(default_factory=list, description="3 bullets")
    decision: Literal["strong_match", "medium_match", "weak_match"]
    advice: list[str] = Field(default_factory=list, description="3 actionable tips to improve the resume for THIS job")

def _build_llm(backend: str = "OFF"):
    backend = (backend or "OFF").upper()

    if backend == "OFF":
        return None

    if backend in ("OPENROUTER", "OPENAI"):
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env")
        return ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=0.2)

    if backend == "OLLAMA":
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        return ChatOllama(model=model, temperature=0.2)

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
        if mode == "cv_job_fit":
            decision = "medium_match" if similarity_score >= 0.45 else "weak_match"
            parsed = ExplainPair(
                explanation=f"LLM disabled. Similarity score={similarity_score:.3f}.",
                strengths=["Semantic similarity suggests overlap."],
                gaps=["LLM reasoning disabled (backend=OFF)."],
                decision=decision,
                advice=["Enable backend=OPENROUTER or backend=OLLAMA to get real advice."]
            ).model_dump()
        else:
            parsed = ExplainRanked(
                explanation=f"LLM disabled. Similarity score={similarity_score:.3f}.",
                strengths=["Semantic similarity suggests overlap."],
                gaps=["LLM reasoning disabled (backend=OFF)."],
            ).model_dump()
        return {"raw_json": "", "parsed": parsed, **parsed}

    if mode == "cv_job_fit":
        schema = ExplainPair
    else:
        schema = ExplainRanked

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
