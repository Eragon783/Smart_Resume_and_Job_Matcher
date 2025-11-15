### Project 4: Smart Resume and Job Matcher

This project focuses on building an AI-powered Resume and Job Matching System that uses embeddings, semantic search, and Generative AI reasoning to match candidates’ resumes with the most relevant job opportunities.

The goal is to go beyond traditional keyword matching by using language understanding models to interpret the meaning, skills, and experience in resumes and job descriptions, enabling contextual, human-like matching.

The system will first parse and process resumes (PDF, DOCX, or text files) to extract structured information such as skills, education, experience, certifications, and interests. Job descriptions (from uploaded files or online sources) will be similarly analyzed. Both resumes and job postings will then be encoded into embeddings using models like Ollama embeddings, SentenceTransformers, or Vertex AI embeddings.

Using these vector representations, the system will compute semantic similarity between candidates and jobs, ranking matches based on contextual relevance rather than exact wording. The AI will also generate explanations for each match, highlighting the reasoning behind compatibility (e.g., “This candidate’s experience in data analytics aligns with the Python and SQL requirements of this role”).