from typing import TypedDict, Dict, Any, Optional

class PipelineState(TypedDict, total=False):
    mode: str
    inputs: Dict[str, Any]
    output: Dict[str, Any]
    error: Optional[str]
