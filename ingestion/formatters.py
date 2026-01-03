# ingestion/json_to_text.py
import os
import json
from tqdm import tqdm


def clean_string(s):
    # Cleaning a value into a single line string
    return str(s).replace("\n", " ").strip()


def json_to_text(data):
    # Converting a structured resume JSON into compact text
    parts = []

    skills = data.get("skills")
    if skills:
        skills_list = [clean_string(s) for s in (skills if isinstance(skills, list) else [skills]) if s]
        parts.append("Skills: " + "; ".join(skills_list))

    experience = data.get("experience")
    if experience:
        exp_list = []
        for e in experience:
            title = clean_string(e.get("title", ""))
            company = clean_string(e.get("company", ""))
            years = clean_string(e.get("years", ""))
            exp_list.append(" ".join(filter(None, [title, "at" if title and company else "", company, years])))
        parts.append("Experience: " + "; ".join(exp_list))

    education = data.get("education")
    if education:
        edu_list = []
        for e in education:
            degree = clean_string(e.get("degree", ""))
            school = clean_string(e.get("school", ""))
            edu_list.append(" ".join(filter(None, [degree, "at" if degree and school else "", school])))
        parts.append("Education: " + "; ".join(edu_list))

    certifications = data.get("certifications")
    if certifications:
        cert_list = [clean_string(c) for c in (certifications if isinstance(certifications, list) else [certifications]) if c]
        parts.append("Certifications: " + "; ".join(cert_list))

    interests = data.get("interests")
    if interests:
        interest_list = [clean_string(i) for i in (interests if isinstance(interests, list) else [interests]) if i]
        parts.append("Interests: " + "; ".join(interest_list))

    summary = data.get("summary")
    if summary:
        parts.append("Summary: " + clean_string(summary))

    return "\n".join(parts).strip()


def process_json_folder(input_folder, output_folder):
    # Converting all JSON files from a folder to .txt files
    os.makedirs(output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]
    for filename in tqdm(json_files, desc="Processing JSON files"):
        input_path = os.path.join(input_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(json_to_text(data))
