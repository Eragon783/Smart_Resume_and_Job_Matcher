import json
import json5
import re
from typing import Any, Tuple

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
