# agents/structuring_agent.py
from __future__ import annotations
import os
import json
from time import sleep
from typing import Optional, List
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


# ---------------------------
# Schema (structured output)
# ---------------------------
class ExperienceItem(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class EducationItem(BaseModel):
    degree: Optional[str] = None
    school: Optional[str] = None
    year: Optional[str] = None


class ResumeSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None

    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)

    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


# ---------------------------
# LLM (Ollama)
# ---------------------------
def build_ollama_llm() -> ChatOllama:
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=model, temperature=0.0)


# ---------------------------
# Prompt + Chain
# ---------------------------
def build_parsing_chain(llm: ChatOllama):
    system = (
        "You are a CV parsing system. "
        "Extract the requested fields from the resume text. "
        "Be faithful to the text: do not invent information. "
        "If a field is not present, return null or an empty list. "
        "Return ONLY structured data."
    )

    user = (
        "Extract these fields from the CV and return structured output:\n"
        "- name\n- email\n- phone\n- location\n- summary\n"
        "- skills (list)\n"
        "- experience (list of objects: title, company, start_date, end_date, description)\n"
        "- education (list of objects: degree, school, year)\n"
        "- certifications (list)\n"
        "- languages (list)\n\n"
        "CV:\n"
        "--------------\n"
        "{cv_text}\n"
        "--------------"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user),
    ])

    # Key part: structured output parsing (no json5 hacks needed)
    structured_llm = llm.with_structured_output(ResumeSchema)

    return prompt, structured_llm


# ---------------------------
# Parsing functions
# ---------------------------
def parse_resume_with_llm(text: str, *, llm: Optional[ChatOllama] = None) -> Optional[dict]:
    llm = llm or build_ollama_llm()
    prompt, structured_llm = build_parsing_chain(llm)

    try:
        out_obj: ResumeSchema = structured_llm.invoke(
            prompt.format_messages(cv_text=text)
        )
        return out_obj.model_dump()
    except Exception as e:
        # Ollama can sometimes fail on very long texts or unusual formats
        print("Parsing error:", repr(e))
        return None


def parse_single_resume(path: str, max_retries: int = 5) -> Optional[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    # Optional: avoid extremely long inputs (can slow/kill local models)
    # You can tune this threshold if needed.
    if len(text) > 50_000:
        text = text[:50_000]

    llm = build_ollama_llm()

    for _ in range(1, max_retries + 1):
        parsed = parse_resume_with_llm(text, llm=llm)
        if parsed:
            return parsed
        sleep(1)

    print(f"Failed parsing after {max_retries} tries: {path}")
    return None


def process_resumes(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".txt")]

    for filename in tqdm(txt_files, desc="Processing CVs", unit="CV"):
        input_path = os.path.join(input_folder, filename)
        output_name = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_folder, output_name)

        if os.path.exists(output_path):
            continue

        parsed = parse_single_resume(input_path)
        if not parsed:
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=4, ensure_ascii=False)
