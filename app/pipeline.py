from __future__ import annotations
from typing import Dict, Any
from io import BytesIO
from app.job_to_cvs_dataset import handle as job_to_cvs_dataset_handler
from app.cv_job_fit import handle as cv_job_fit_handler
from app.cv_to_jobs import handle as cv_to_jobs_handler
from app.job_to_cvs import handle as job_to_cvs_upload_handler
from app.cv_to_jobs_dataset import handle as cv_to_linkedin_jobs_dataset_handler
import numpy as np

PIPELINE_VERSION = "v3_reuse_loaders_matching_explainer"

# ---------------------------
# Router
# ---------------------------

def run(mode: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if mode == "job_to_cvs_dataset":
        out = job_to_cvs_dataset_handler(inputs)
        out["pipeline_version"] = PIPELINE_VERSION
        return out

    if mode == "cv_job_fit":
        out = cv_job_fit_handler(inputs)
        out["pipeline_version"] = PIPELINE_VERSION
        return out
    
    if mode == "cv_to_jobs":
        out = cv_to_jobs_handler(inputs)
        out["pipeline_version"] = PIPELINE_VERSION
        return out
    
    if mode == "job_to_cvs_upload":
        out = job_to_cvs_upload_handler(inputs)
        out["pipeline_version"] = PIPELINE_VERSION
        return out
    
    if mode == "cv_to_linkedin_jobs_dataset":
        out = cv_to_linkedin_jobs_dataset_handler(inputs)
        out["pipeline_version"] = PIPELINE_VERSION
        return out


    return {
        "status": "NOT_IMPLEMENTED_YET",
        "mode": mode,
        "inputs_received": list(inputs.keys()),
        "pipeline_version": PIPELINE_VERSION,
    }