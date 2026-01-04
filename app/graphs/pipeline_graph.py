from langgraph.graph import StateGraph, END
from .state import PipelineState
from . import nodes

def _route(state: PipelineState) -> str:
    mode = state.get("mode", "")
    mapping = {
        "job_to_cvs_dataset": "job_to_cvs_dataset",
        "cv_job_fit": "cv_job_fit",
        "cv_to_jobs": "cv_to_jobs",
        "job_to_cvs_upload": "job_to_cvs_upload",
        "cv_to_linkedin_jobs_dataset": "cv_to_linkedin_jobs_dataset",
    }
    return mapping.get(mode, "not_implemented")

def build_pipeline_graph():
    g = StateGraph(PipelineState)

    g.add_node("job_to_cvs_dataset", nodes.node_job_to_cvs_dataset)
    g.add_node("cv_job_fit", nodes.node_cv_job_fit)
    g.add_node("cv_to_jobs", nodes.node_cv_to_jobs)
    g.add_node("job_to_cvs_upload", nodes.node_job_to_cvs_upload)
    g.add_node("cv_to_linkedin_jobs_dataset", nodes.node_cv_to_linkedin_jobs_dataset)
    g.add_node("not_implemented", nodes.node_not_implemented)

    # entry is a conditional router
    g.set_conditional_entry_point(
        _route,
        {
            "job_to_cvs_dataset": "job_to_cvs_dataset",
            "cv_job_fit": "cv_job_fit",
            "cv_to_jobs": "cv_to_jobs",
            "job_to_cvs_upload": "job_to_cvs_upload",
            "cv_to_linkedin_jobs_dataset": "cv_to_linkedin_jobs_dataset",
            "not_implemented": "not_implemented",
        },
    )

    # every node ends the workflow
    g.add_edge("job_to_cvs_dataset", END)
    g.add_edge("cv_job_fit", END)
    g.add_edge("cv_to_jobs", END)
    g.add_edge("job_to_cvs_upload", END)
    g.add_edge("cv_to_linkedin_jobs_dataset", END)
    g.add_edge("not_implemented", END)

    return g.compile()
