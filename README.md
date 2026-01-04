> :warn: You need to download [jobs_index.faiss](https://devinci-my.sharepoint.com/personal/sarah_ounes_edu_devinci_fr/_layouts/15/guestaccess.aspx?share=IQAZta4fxG5oS6-ZUeO0vV_4AV56LaHPKu4snt0XCqS0k0c&e=mrp7Rg) and [jobs_index_mapping.json](https://devinci-my.sharepoint.com/personal/sarah_ounes_edu_devinci_fr/_layouts/15/guestaccess.aspx?share=IQBzqWv453dXQLbCvYgcOth4Af7FRNZ33Hms8iGNrjI3m0U&e=rUF7Le) and put them in "data/job_treated/" to run the project.

### Project 4: Smart Resume and Job Matcher

This project focuses on building an AI-powered Resume and Job Matching System that uses embeddings, semantic search, and Generative AI reasoning to match candidates’ resumes with the most relevant job opportunities.

The goal is to go beyond traditional keyword matching by using language understanding models to interpret the meaning, skills, and experience in resumes and job descriptions, enabling contextual, human-like matching.

The system will first parse and process resumes (PDF, DOCX, or text files) to extract structured information such as skills, education, experience, certifications, and interests. Job descriptions (from uploaded files or online sources) will be similarly analyzed. Both resumes and job postings will then be encoded into embeddings using models like Ollama embeddings, SentenceTransformers, or Vertex AI embeddings.

Using these vector representations, the system will compute semantic similarity between candidates and jobs, ranking matches based on contextual relevance rather than exact wording. The AI will also generate explanations for each match, highlighting the reasoning behind compatibility (e.g., “This candidate’s experience in data analytics aligns with the Python and SQL requirements of this role”).

Datatsets for resume : https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset | https://github.com/NataliaVanetik/vacancy-resume-matching-dataset/tree/main/CV

Dataset for job offer : https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset


# Smart Resume and Job Matcher

This repository contains a **Generative AI system** that matches resumes to job offers using a structured, reproducible pipeline. The project relies on a remote Large Language Model (LLM) for one specific step (resume text → structured JSON) and is designed to run consistently on **Windows, Linux, and macOS**.

---

## Project Overview

The system performs the following steps:

1. Extract raw text from resumes (PDF, DOCX, TXT)
2. Use an LLM to convert resume text into structured JSON
3. Prepare structured text for embeddings
4. Match resumes to job offers using similarity scoring
5. Provide notebooks for demonstration, evaluation, and reproducibility

Only step 2 requires access to a remote LLM API.

---

## Project Structure

Smart_Resume_and_Job_Matcher  
├── .env                  (local file, not committed)  
├── .gitignore  
├── README.md  
├── requirements.txt  
├── agents/               (LLM-related logic: structuring, matching)  
├── ingestion/            (loaders, cleaners, embeddings)  
├── notebooks/            (execution and evaluation notebooks)  
├── data/                 (resumes, jobs, generated outputs)  
└── app/                  (pipeline orchestration, optional)

---

## Requirements

- Python 3.9 or newer
- pip

After cloning the repository, install dependencies with:

pip install -r requirements.txt

---



## Environment Setup

### Windows
powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## LLM API Setup and Reproducibility Guide

This project uses a **remote LLM API** only to transform raw resume text into structured JSON (Notebook 2). Because the LLM is accessed over the Internet, **each user must provide their own API key**. The key is never included in the repository. This is standard practice in academic and industrial GenAI projects.

To avoid operating-system-specific issues and ensure reproducibility, the API key is loaded from a **local `.env` file**.

---

## What Is an API Key and Why Is It Needed?

An API key is a personal secret token that authorizes your machine to call a remote LLM service.

- It works like a password
- It identifies who is making the request
- It must never be shared publicly
- It must never be committed to GitHub

The API key is required only for the resume structuring step. All other parts of the pipeline run locally.

---

## Step 1 – Clone the Repository

Clone the repository and move into the project directory:

git clone <REPOSITORY_URL>  
cd Smart_Resume_and_Job_Matcher

All following steps must be executed from the project root directory.

---

## Step 2 – Create a Local `.env` File

At the root of the project, create a file named `.env`.

This file exists only on your local machine and is ignored by Git via `.gitignore`.

---

## Step 3 – Create an API Key

You can use either of the following providers:

OpenRouter (recommended):  
- Go to https://openrouter.ai  
- Create an account  
- Open the API Keys section  
- Create a new API key  
- Copy the key (usually starts with sk-or-)

OpenAI:  
- Go to https://platform.openai.com/api-keys  
- Create an API key  
- Copy the key (usually starts with sk-)

---

## Step 4 – Add the API Key to the `.env` File

Open the `.env` file and add exactly one line:

OPENAI_API_KEY=sk-or-xxxxxxxxxxxxxxxx

Rules:
- No quotes
- No spaces
- One line only
- Never commit this file

---

## Step 5 – How the API Key Is Used

The project uses the python-dotenv library to automatically load the `.env` file at runtime.

- The key is read using os.getenv("OPENAI_API_KEY")
- The key is never hardcoded
- The key is never stored in the repository

This approach works reliably on Windows, Linux, and macOS, including when running notebooks in VS Code or Jupyter.

---

## Step 6 – Verify the Configuration

Before running Notebook 2, verify that the API key is correctly loaded.

In a Python shell or notebook cell:

import os  
print(os.getenv("OPENAI_API_KEY"))

If the key is printed, the configuration is correct. If None is printed, the `.env` file is missing or incorrectly configured.

---

## Vector Index Backend (FAISS vs Chroma)

This project supports **two interchangeable vector index backends** for resume embeddings:
**FAISS** and **Chroma**.

Both backends are fully compatible with the pipeline and can be selected
at runtime without changing the core logic of the project.

The goal is to ensure:
- cross-platform compatibility (Windows / Linux / macOS)
- reproducibility for all users
- flexibility between performance-oriented and persistence-oriented setups

---

## Available Backends

### FAISS (Facebook AI Similarity Search)

FAISS is a high-performance vector similarity library optimized for fast
nearest-neighbor search.

**Characteristics:**
- Very fast similarity search
- In-memory index
- Ideal for performance benchmarks and large-scale experiments
- Widely used in research and industry

**Limitations:**
- Installation can be problematic on some Windows environments
- Requires manual saving/loading of the index

**Files produced:**
- `data/resume_index.faiss`
- `data/resume_index_mapping.json`

---

### Chroma

Chroma is a lightweight vector database designed for LLM and RAG applications.

**Characteristics:**
- Persistent on disk by default
- Easy to install on all platforms
- Supports metadata (e.g. filenames)
- Well suited for RAG-style pipelines and demos

**Limitations:**
- Slightly slower than FAISS for pure similarity search
- Less optimized for very large-scale indexing

**Files produced:**
- `data/chroma_resume_db/`
- `data/resume_index_mapping.json`

---

## Why Two Backends?

Supporting both FAISS and Chroma provides important benefits:

- **Reproducibility**: if FAISS cannot be installed on a system, Chroma can be used instead
- **Cross-platform compatibility**: Chroma works reliably on Windows, Linux, and macOS
- **Flexibility**: FAISS for performance experiments, Chroma for persistence and RAG workflows
- **Pedagogical value**: illustrates different vector indexing strategies

The rest of the pipeline (LLM structuring, embedding model, matching logic)
remains identical regardless of the backend.

---

## How to Select the Backend

The backend is selected by changing a single variable:

```python
BACKEND = "faiss"    # or "chroma"

