import os
import re
from tqdm import tqdm
import docx
from pdfminer.high_level import extract_text


def read_txt(file_path):
    # Reading a plain text file
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx(file_path):
    # Reading all paragraphs from a DOCX file
    return "\n".join([p.text for p in docx.Document(file_path).paragraphs])


def read_pdf(file_path):
    # Extracting text from a PDF file
    return extract_text(file_path)


def extract_text(file_path):
    # Choosing the correct reader based on file extension
    file_path_lower = file_path.lower()
    if file_path_lower.endswith(".pdf"):
        return read_pdf(file_path)
    if file_path_lower.endswith(".docx"):
        return read_docx(file_path)
    if file_path_lower.endswith(".txt"):
        return read_txt(file_path)


def clean_text(text):
    # Removing control characters and normalizing whitespace
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_all_texts(input_folder, output_folder):
    # Supported document formats
    supported_extensions = (".pdf", ".docx", ".txt")

    # Creating output directory if needed
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder), desc="Extracting files"):
        # Skipping unsupported files
        if not filename.lower().endswith(supported_extensions):
            continue

        input_path = os.path.join(input_folder, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(output_folder, output_filename)

        # Skipping already processed files
        if os.path.exists(output_path):
            continue

        # Extracting and cleaning text
        text = clean_text(extract_text(input_path))

        # Saving cleaned text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
