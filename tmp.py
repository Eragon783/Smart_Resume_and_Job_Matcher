from app.pipeline import run
import pathlib

resume = {
    'filename':'10553553.pdf',
    'bytes':pathlib.Path('/home/lllm/Documents/SH/Smart_Resume_and_Job_Matcher/data/resume/10553553.pdf').read_bytes()
}
job={
    'filename':'sample_job.txt',
    'bytes':pathlib.Path('/home/lllm/Documents/SH/Smart_Resume_and_Job_Matcher/data/job_test/sample_job.txt').read_bytes()
}

out = run('cv_job_fit', {'resume_file': resume,'job_offer_file': job,'add_explanations':True})
print(out.get('status'), out.get('similarity_score'))