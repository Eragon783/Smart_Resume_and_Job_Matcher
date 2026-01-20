# Smart Resume and Job Matcher (Project 4)

This project implements a **Smart Resume and Job Matching System** based on **Generative AI**, semantic embeddings, and vector similarity search.  
Its objective is to match resumes to job offers in a contextual and human-like manner, going beyond traditional keyword-based approaches.

The system is designed to be **reproducible, modular, and cross-platform**, running consistently on **Windows, Linux, and macOS**.

Only one step of the pipeline relies on a remote Large Language Model (LLM); all other components operate locally.

---

## Project Objectives

The main goals of this project are:

- Automatically extract meaningful information from resumes
- Structure unformatted resume text into a clean JSON representation
- Represent resumes and job offers using semantic embeddings
- Perform similarity-based matching between candidates and job descriptions
- Provide explainable results highlighting strengths, gaps, and match quality
- Ensure reproducibility and portability across different operating systems

---

## Global Pipeline Overview

The system follows a clear and structured pipeline:

1. Resume files are parsed and their raw text is extracted  
2. Raw resume text is transformed into a structured JSON format using an LLM  
3. Structured data is converted into a textual representation suitable for embeddings  
4. Semantic embeddings are computed for resumes and job offers  
5. A vector similarity search ranks the most relevant matches  
6. The system generates a human-readable explanation for each match  

Only the resume structuring step requires an external API.  
Embedding, indexing, matching, and explanation logic run locally.

---

## Datasets Used

This project relies on publicly available datasets for experimentation and evaluation.

### Resume datasets
- Kaggle Resume Dataset  
  https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset  
- Vacancy–Resume Matching Dataset  
  https://github.com/NataliaVanetik/vacancy-resume-matching-dataset/tree/main/CV  

### Job offer dataset
- Job Description Dataset  
  https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024 

Some datasets and derived CSV files are not included in the repository due to storage constraints and must be kept locally.

---

## Large Files and External Resources

To keep the repository lightweight and compliant with GitHub storage limits, some large artifacts are not stored directly in the repository.

The following files **must be downloaded manually**:

- jobs_index.faiss  
- jobs_index_mapping.json  

These files must be placed in the following directory:

- data/job_treated/

Download links:
- https://devinci-my.sharepoint.com/personal/sarah_ounes_edu_devinci_fr/_layouts/15/guestaccess.aspx?share=IQAZta4fxG5oS6-ZUeO0vV_4AV56LaHPKu4snt0XCqS0k0c&e=mrp7Rg  
- https://devinci-my.sharepoint.com/personal/sarah_ounes_edu_devinci_fr/_layouts/15/guestaccess.aspx?share=IQBzqWv453dXQLbCvYgcOth4Af7FRNZ33Hms8iGNrjI3m0U&e=rUF7Le  

This approach reflects standard practice in machine learning projects where large indexes and datasets cannot be versioned with Git.

---

## Use of Ollama (Local Models)

The project optionally supports **Ollama** to run local language models and generate embeddings without relying on external services.

Ollama can be used for:
- Local chat-based models
- Local embedding generation

Using Ollama improves privacy, reproducibility, and offline experimentation.  
The exact model choice depends on the user’s configuration and available system resources.

---

## Use of Remote LLM APIs

A remote LLM API is used **only** for transforming raw resume text into a structured JSON representation.

Each user must provide their own API key.  
The key is stored locally and never included in the repository.

This design ensures:
- Clear separation between local and remote components
- Reproducibility across machines
- Compliance with security best practices

---

## API Key Management and Reproducibility

API keys are stored in a local environment file that is ignored by version control.

Key principles:
- The API key is personal and confidential
- It must never be committed to GitHub
- It is required only for the resume structuring step
- All other steps run independently of the API

This approach guarantees consistent behavior across operating systems and development environments.

---

## Vector Indexing Backends

The project supports two interchangeable vector indexing backends:

### FAISS
FAISS is optimized for high-performance nearest-neighbor search and is commonly used in large-scale research and industrial systems.

Strengths:
- Very fast similarity search
- Suitable for large embedding collections

Limitations:
- Installation may be more complex on some systems
- Index persistence must be handled manually

### Chroma
Chroma is a lightweight vector database designed for LLM and RAG pipelines.

Strengths:
- Persistent storage by default
- Easy installation across platforms
- Supports metadata management

Limitations:
- Slightly slower than FAISS for pure similarity search

Both backends produce equivalent results and can be swapped without modifying the core logic of the project.

---

## Backend Selection

The vector backend can be changed through a configuration variable in the codebase.

This flexibility allows:
- Performance-oriented experiments using FAISS
- Persistence-oriented workflows using Chroma
- Guaranteed reproducibility when one backend is unavailable

---

## Cross-Platform Design

Special care has been taken to ensure that the project runs consistently on:
- Windows
- Linux
- macOS

This includes:
- Environment-based configuration
- Avoidance of OS-specific paths
- Explicit handling of external resources

---

## Summary

This project demonstrates how Generative AI, semantic embeddings, and vector databases can be combined to build an intelligent, explainable resume–job matching system.

It emphasizes:
- Modularity and clarity
- Reproducibility
- Responsible use of external APIs
- Practical constraints of real-world ML projects

The resulting system provides a strong foundation for further extensions such as large-scale deployment, advanced ranking strategies, or full RAG-based recruitment pipelines.
