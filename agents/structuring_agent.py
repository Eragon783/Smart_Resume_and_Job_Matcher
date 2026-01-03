# agents/structuring_agent.py
import os
import json
import json5
from time import sleep
from openai import OpenAI
from tqdm import tqdm


# Creating OpenAI client (OpenRouter)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)


def safe_json_parse(content):
    # Parsing JSON safely (json5 tolerates small formatting issues)
    try:
        return json5.loads(content)
    except Exception as e:
        print("Malformed JSON:", e)
        return None


def parse_resume_with_llm(text):
    # Building a strict prompt to get structured CV data
    prompt = f"""
    You are a CV parsing system.
    Extract the following fields from the CV below and return STRICTLY valid JSON:
    - name
    - email
    - phone
    - location
    - summary
    - skills (list)
    - experience (list of objects: title, company, start_date, end_date, description)
    - education (list of objects: degree, school, year)
    - certifications (list)
    - languages (list)

    CV:
    --------------
    {text}
    --------------
    
    Reply ONLY with valid JSON.
    """

    response = client.chat.completions.create(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )

    return safe_json_parse(response.choices[0].message.content)


def parse_single_resume(path, max_retries=10):
    # Loading a single extracted resume text file
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    # Retrying in case LLM returns malformed JSON
    for _ in range(1, max_retries + 1):
        parsed = parse_resume_with_llm(text)
        if parsed:
            return parsed
        sleep(1)

    print(f"Failed parsing after {max_retries} tries : {path}")
    return None


def process_resumes(input_folder, output_folder):
    # Processing all .txt resumes in a folder into .json files
    os.makedirs(output_folder, exist_ok=True)

    txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".txt")]

    for filename in tqdm(txt_files, desc="Processing CVs", unit="CV"):
        input_path = os.path.join(input_folder, filename)
        output_name = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_folder, output_name)

        # Skipping if already processed
        if os.path.exists(output_path):
            continue

        parsed = parse_single_resume(input_path)
        if not parsed:
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=4, ensure_ascii=False)
