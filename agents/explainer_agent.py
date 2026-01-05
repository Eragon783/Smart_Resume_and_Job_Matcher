from __future__ import annotations
from typing import Literal, Optional, Dict, Any
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import re
import json5
import prompt

def safe_json_parse(text: str) -> tuple[dict, str | None]:
    if text is None:
        return {}, "raw_text_is_none"
    raw = str(text).strip()
    if not raw:
        return {}, "raw_text_is_empty"

    # remove ```json fences
    raw2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    # try json
    try:
        return json.loads(raw2), None
    except Exception:
        pass

    # try json5 (tolerant)
    try:
        return json5.loads(raw2), None
    except Exception:
        pass

    # extract first {...}
    m = re.search(r"\{.*\}", raw2, flags=re.DOTALL)
    if m:
        chunk = m.group(0)
        try:
            return json.loads(chunk), None
        except Exception:
            try:
                return json5.loads(chunk), None
            except Exception as e:
                return {}, f"extracted_json_parse_failed: {type(e).__name__}"

    return {}, "no_json_object_found"
#########################################################################


load_dotenv()

Mode = Literal["job_to_resumes", "resume_to_jobs", "pair_compatibility"]

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

    if mode in ("job_to_resumes", "pair_compatibility") and not job_text:
        raise ValueError("job_text is required for this mode")
    if mode in ("resume_to_jobs", "pair_compatibility") and not resume_text:
        raise ValueError("resume_text is required for this mode")

    llm = _build_llm(backend)

    if llm is None:
        if mode == "pair_compatibility":
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

    # --- Prompt (better, more “human-like”)
    system = (
        "You are a recruitment assistant. "
        "Be concrete: refer to specific skills/experiences mentioned. "
        "Return ONLY valid JSON matching the required schema. No markdown."
    )

    if mode == "pair_compatibility":
        schema = ExplainPair
        task = (
            "Task:\n"
            "1) explanation (4-6 lines)\n"
            "2) strengths: 3 bullets\n"
            "3) gaps: 3 bullets\n"
            "4) decision: strong_match|medium_match|weak_match\n"
            "5) advice: 3 actionable improvements for the resume to better fit THIS job\n"
        )
    else:
        schema = ExplainRanked
        task = (
            "Task:\n"
            "1) explanation (3-5 lines)\n"
            "2) strengths: 3 bullets\n"
            "3) gaps: 2 bullets\n"
        )

    user = (
        f"{task}\n"
        "Context:\n"
        "- rank: {rank}\n"
        "- similarity_score: {score}\n\n"
        "JOB OFFER:\n{job}\n\n"
        "RESUME:\n{resume}\n"
    )

    backend_u = (backend or "OFF").upper()

    # ✅ OpenAI / OpenRouter : tu peux garder structured output
    if backend_u in ("OPENAI", "OPENROUTER"):
        structured = llm.with_structured_output(schema)
        out_obj = structured.invoke(
            prompt.format_messages(
                rank=top_k_rank if top_k_rank is not None else "",
                score=f"{similarity_score:.3f}",
                job=job_text or "",
                resume=resume_text or "",
            )
        )
        parsed = out_obj.model_dump()
        return {
            "raw_json": json.dumps(parsed, ensure_ascii=False),
            "parsed": parsed,
            **parsed
        }

    # ✅ Ollama : PAS de with_structured_output → prompt JSON + parse
    # On demande un JSON strict (sans markdown)
    json_instruction = (
        "Return ONLY a valid JSON object with the EXACT keys required by the schema. "
        "No extra text, no markdown, no explanations outside JSON."
    )

    messages = prompt.format_messages(
        rank=top_k_rank if top_k_rank is not None else "",
        score=f"{similarity_score:.3f}",
        job=job_text or "",
        resume=resume_text or "",
    )

    # On injecte l’instruction JSON dans le dernier message utilisateur
    messages[-1].content = json_instruction + "\n\n" + messages[-1].content

    raw_text = llm.invoke(messages).content  # <- Ollama renvoie du texte
    parsed, err = safe_json_parse(raw_text)

    # fallback minimal si parsing échoue
    if err or not isinstance(parsed, dict) or not parsed:
        parsed = {
            "explanation": raw_text.strip()[:1200] if raw_text else "No LLM output.",
            "strengths": [],
            "gaps": [],
        }
        if mode == "pair_compatibility":
            parsed.update({"decision": "weak_match", "advice": []})
        err = err or "invalid_or_empty_json"

    return {
        "raw_json": raw_text,
        "parsed": parsed,
        "parse_error": err,
        **parsed
    }
