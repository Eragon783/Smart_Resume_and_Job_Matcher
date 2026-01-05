import json
import re
import requests # type: ignore
import json5 # type: ignore

def safe_json_parse(text: str):
    if text is None:
        return {}, "raw_text_is_none"
    raw = str(text).strip()
    if not raw:
        return {}, "raw_text_is_empty"

    raw2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        return json.loads(raw2), None
    except Exception:
        pass

    try:
        return json5.loads(raw2), None
    except Exception:
        pass

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
