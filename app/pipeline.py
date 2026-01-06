from __future__ import annotations
from typing import Dict, Any
from app.graphs.pipeline_graph import build_pipeline_graph



_GRAPH = None

def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_pipeline_graph()
    return _GRAPH

def run(mode: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint called by Streamlit:
    out = run(mode, inputs)

    Always routes through LangGraph.
    """
    try:
        g = _get_graph()

        final_state = g.invoke({
            "mode": mode,
            "inputs": inputs,
        })

        out = final_state.get("output")
        if isinstance(out, dict):
            return out

        # If graph didn't set output, surface a clear error
        return {
            "status": "ERROR",
            "mode": mode,
            "error": final_state.get("error") or "Graph finished without output",
            "debug_state_keys": list(final_state.keys()) if isinstance(final_state, dict) else None,
        }

    except Exception as e:
        return {"status": "ERROR", "mode": mode, "error": repr(e)}

