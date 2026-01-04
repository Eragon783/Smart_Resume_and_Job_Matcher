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
    graph = _get_graph()
    out = graph.invoke({"mode": mode, "inputs": inputs})
    return out["output"]
