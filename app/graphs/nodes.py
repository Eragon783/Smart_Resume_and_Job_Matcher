from typing import Dict, Any
from .state import PipelineState

from app.job_to_cvs_dataset import handle as job_to_cvs_dataset_handler
from app.cv_job_fit import handle as cv_job_fit_handler
from app.cv_to_jobs import handle as cv_to_jobs_handler
from app.job_to_cvs import handle as job_to_cvs_upload_handler
from app.cv_to_jobs_dataset import handle as cv_to_linkedin_jobs_dataset_handler

PIPELINE_VERSION = "vf_langgraph"

def _finalize(out: Dict[str, Any]) -> Dict[str, Any]:
    out["pipeline_version"] = PIPELINE_VERSION
    return out

def node_job_to_cvs_dataset(state: PipelineState) -> PipelineState:
    out = job_to_cvs_dataset_handler(state["inputs"])
    return {"output": _finalize(out)}

def node_cv_job_fit(state: PipelineState) -> PipelineState:
    out = cv_job_fit_handler(state["inputs"])
    return {"output": _finalize(out)}

def node_cv_to_jobs(state: PipelineState) -> PipelineState:
    out = cv_to_jobs_handler(state["inputs"])
    return {"output": _finalize(out)}

def node_job_to_cvs_upload(state: PipelineState) -> PipelineState:
    out = job_to_cvs_upload_handler(state["inputs"])
    return {"output": _finalize(out)}

def node_cv_to_linkedin_jobs_dataset(state: PipelineState) -> PipelineState:
    out = cv_to_linkedin_jobs_dataset_handler(state["inputs"])
    return {"output": _finalize(out)}

def node_not_implemented(state: PipelineState) -> PipelineState:
    mode = state.get("mode", "unknown")
    inputs = state.get("inputs", {})
    return {
        "output": {
            "status": "NOT_IMPLEMENTED_YET",
            "mode": mode,
            "inputs_received": list(inputs.keys()),
            "pipeline_version": PIPELINE_VERSION,
        }
    }
