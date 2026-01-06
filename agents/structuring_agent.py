# agents/structuring_agent.py
from __future__ import annotations
import os
import json
from time import sleep
from typing import Optional, List
from tqdm import tqdm # type: ignore
from pydantic import BaseModel, Field # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_community.chat_models import ChatOllama # type: ignore
import streamlit as st # type: ignore
import pandas as pd

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

def normalize_llm_payload(llm_any):
    """
    Normalize LLM outputs for DISPLAY purposes only.
    We only accept explanation-style outputs.
    """
    if llm_any is None:
        return None

    # Case 1: agent already returned parsed explanation
    if isinstance(llm_any, dict):
        if "explanation" in llm_any:
            return llm_any

        # If wrapped format {raw_json, parsed}
        parsed = llm_any.get("parsed")
        if isinstance(parsed, dict) and "explanation" in parsed:
            return parsed

        return None  # 🚫 reject CV parsing outputs

    # Case 2: raw string → try extract JSON
    if isinstance(llm_any, str) and llm_any.strip():
        parsed = _extract_json_from_markdown(llm_any)
        if isinstance(parsed, dict) and "explanation" in parsed:
            return parsed

    return None



def render_hits_table(hits):
    if not hits:
        st.info("No results.")
        return

    rows = []
    for h in hits:
        rank = h.get("rank")
        score = h.get("score")
        url = h.get("url")
        title = h.get("title")
        filename = h.get("filename")
        index_id = h.get("index_id")

        label = title or filename or url or (f"item_{index_id}" if index_id is not None else "item")

        row = {
            "rank": rank,
            "label": label,
            "score": score,
        }

        # include url only if present (LinkedIn mode)
        if url:
            row["url"] = url

        # keep old compatibility info if present
        if filename and not title:
            row["filename"] = filename

        rows.append(row)

    df = pd.DataFrame(rows) # type: ignore

    # Choosing a stable column order based on availability
    cols = ["rank", "label", "score"]
    if "url" in df.columns:
        cols.insert(2, "url")  # rank, label, url, score
    if "filename" in df.columns and "url" not in df.columns:
        cols.insert(2, "filename")

    # Keeping any extra columns (rare) at the end
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]

    st.dataframe(df[cols], use_container_width=True)


def render_hits_cards(hits):

    if not hits:
        st.info("No results.")
        return

    for h in hits:
        rank = h.get("rank")
        score = h.get("score")
        title = h.get("title")
        filename = h.get("filename")
        url = h.get("url")
        index_id = h.get("index_id")

        label = title or filename or url or (f"item_{index_id}" if index_id is not None else "item")

        with st.container(border=True):
            if isinstance(score, (int, float)):
                st.markdown(f"**N°{rank} — {label}**  \nScore: `{score:.4f}`")
            else:
                st.markdown(f"**N°{rank} — {label}**")

            # LinkedIn: clickable link if present
            if url:
                st.markdown(f"[Open on LinkedIn]({url})")

            if filename and label != filename:
                st.caption(f"File: {filename}")

            llm_final = normalize_llm_payload(h.get("llm_explanation"))

            if llm_final:
                with st.expander("LLM decision"):
                    render_llm_explanation_pretty(llm_final)
            elif h.get("llm_explanation") is not None:
                # If something exists but isn't parseable -> show raw for debugging
                with st.expander("LLM raw output (unparsed)"):
                    st.write(h.get("llm_explanation"))


            # Debug always available
            with st.expander("Debug (raw hit)"):
                st.json(h)

def label_from_score(score: float) -> str:
    if score >= 0.60:
        return "Strong fit"
    if score >= 0.30:
        return "Moderate fit"
    return "Weak fit"

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return [str(x)]

def _extract_json_from_markdown(s: str) -> dict | None:
    """
    Extracting JSON that may be wrapped like:
    ```json
    {...}
    ```
    """
    if not s or not isinstance(s, str):
        return None

    # Try to find a ```json ... ``` block first
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL) # type: ignore
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: try to parse the whole string as JSON
    s2 = s.strip()
    try:
        return json.loads(s2)
    except Exception:
        return None

def render_cv_job_fit(out: dict):
    st.markdown("## Compatibility result")

    score = float(out.get("similarity_score", 0.0))
    st.metric("Cosine similarity", f"{score:.4f}", label_from_score(score))

    llm = out.get("llm_explanation") or out.get("llm")

    # ✅ Cas 1: aucun output LLM
    if llm is None:
        st.warning("LLM did not return any output.")
        with st.expander("Diagnostics", expanded=True):
            st.json(out.get("diagnostics", {}))
        return

    llm_final = normalize_llm_payload(llm)
    if not llm_final:
        st.warning("LLM output ignored (invalid explanation format).")
        with st.expander("Raw LLM output"):
            st.write(llm)
        return


    # ✅ Cas 2: output présent mais non parsable en dict
    if not isinstance(llm_final, dict):
        st.warning("LLM returned output but it could not be parsed into JSON.")
        with st.expander("LLM raw output", expanded=True):
            st.write(llm)
        with st.expander("Diagnostics", expanded=False):
            st.json(out.get("diagnostics", {}))
        return

    # ✅ Cas 3: dict OK, mais parse_error disponible (si ton agent l'ajoute)
    parse_error = llm_final.get("parse_error") or (llm.get("parse_error") if isinstance(llm, dict) else None)
    raw_json = (llm.get("raw_json") if isinstance(llm, dict) else None)

    if parse_error:
        st.warning(f"LLM JSON parsing warning: {parse_error}")
        if raw_json:
            with st.expander("LLM raw output", expanded=False):
                st.write(raw_json)


    decision = llm_final.get("decision", "N/A")
    explanation = llm_final.get("explanation", "")

    strengths = llm_final.get("strengths", [])
    gaps = llm_final.get("gaps", [])
    advice = llm_final.get("advice") or llm_final.get("recommendations") or llm_final.get("improvements")

    st.markdown("### LLM decision")
    st.write("**Decision:**", decision)
    if explanation:
        st.write("**Explanation:**")
        st.write(explanation)

    st.markdown("### Strengths")
    s_list = _as_list(strengths)
    if s_list:
        for s in s_list:
            st.write(f"- {s}")
    else:
        st.write("- (none)")

    st.markdown("### Gaps")
    g_list = _as_list(gaps)
    if g_list:
        for g in g_list:
            st.write(f"- {g}")
    else:
        st.write("- (none)")

    st.markdown("### Advice to improve your CV for this job")
    a_list = _as_list(advice) if advice is not None else []
    if a_list:
        for a in a_list:
            st.write(f"- {a}")
    else:
        # fallback: derive from gaps
        if g_list:
            for g in g_list:
                st.write(f"- Add or strengthen: {g}")
        else:
            st.write("- Add more job-specific keywords and measurable achievements.")

    with st.expander("LLM raw output (parsed)", expanded=False):
        st.json(llm_final)

    with st.expander("Diagnostics", expanded=False):
        st.json(out.get("diagnostics", {}))
        

def render_llm_explanation_pretty(llm: dict):
    """
    Pretty display for LLM output across ALL modes.
    Renders: Decision, Explanation, Strengths, Gaps
    Does NOT render Advice (by design).
    """
    if not isinstance(llm, dict):
        st.warning("LLM explanation is not a dict.")
        st.write(llm)
        return

    # Flexible key support (depending on your agent outputs)
    decision = llm.get("decision") or llm.get("match_decision") or llm.get("label")
    explanation = llm.get("explanation") or llm.get("reason") or llm.get("summary")

    strengths = llm.get("strengths") or llm.get("pros") or llm.get("positive_points")
    gaps = llm.get("gaps") or llm.get("missing") or llm.get("cons") or llm.get("negative_points")

    st.markdown("## LLM decision")

    if decision:
        st.markdown(f"**Decision:** `{decision}`")

    if explanation:
        st.markdown("**Explanation:**")
        st.write(explanation)

    if strengths:
        st.markdown("## Strengths")
        for s in _as_list(strengths):
            st.markdown(f"- {s}")

    if gaps:
        st.markdown("## Gaps")
        for g in _as_list(gaps):
            st.markdown(f"- {g}")

    with st.expander("LLM raw output (parsed)"):
        st.json(llm)


