# utils/json_utils.py

import json
import re

def safe_json_parse(raw: str) -> dict:
    raw = raw.strip()

    # Trying direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Trying to extract JSON block
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Fallback
    return {"explanation": raw, "strengths": [], "gaps": []}
