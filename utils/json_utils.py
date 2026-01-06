from __future__ import annotations
import json
import json5
import re
from typing import Any, Tuple
import requests
import re
from typing import Optional
import io
from pypdf import PdfReader  # or pdfplumber if you 

def safe_json_parse(text: str) -> Tuple[Any, str | None]:
    if not text or not isinstance(text, str):
        return None, "empty_text"

    s = text.strip()

    # Removing ```json fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()

    # Extracting the largest JSON object candidate
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_object_found"

    candidate = s[start:end+1].strip()

    # Trying strict JSON first, then json5
    try:
        return json.loads(candidate), None
    except Exception:
        try:
            return json5.loads(candidate), None
        except Exception as e:
            return None, f"extracted_json_parse_failed: {type(e).__name__}"


def ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # Ollama /api/chat returns: {"message": {"role": "...", "content": "..."}, ...}
    return (data.get("message") or {}).get("content", "").strip()

def sanitize_text_for_llm(text: str) -> str:
    # Removing weird whitespace / control chars
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def smart_trim(text: str, max_chars: int = 12000) -> str:
    # Trimming while keeping head+tail (often best for CVs/jobs)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()

from typing import Any, Dict, Union, Tuple
import io

def extract_text_from_file(file_or_name: Union[Dict[str, Any], str], b: bytes = None) -> str:
    """
    Accepts either:
      - file dict: {"filename": "...", "bytes": b"..."}
      - (filename: str, bytes: bytes)

    This avoids breaking existing code and fixes your current handle().
    """
    # --- Normalizing inputs ---
    if isinstance(file_or_name, dict):
        filename = (file_or_name.get("filename") or "")
        data = file_or_name.get("bytes") or b""
    else:
        filename = file_or_name or ""
        data = b or b""

    ext = (filename.split(".")[-1] if "." in filename else "").lower()

    if ext == "txt":
        return data.decode("utf-8", errors="ignore").strip()

    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages).strip()

    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    return data.decode("utf-8", errors="ignore").strip()
