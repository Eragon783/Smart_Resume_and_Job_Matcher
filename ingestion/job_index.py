import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss  # faiss-cpu

def _read_txt_files(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".txt")])
    paths = [os.path.join(folder, f) for f in files]
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return files, texts

def _encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int):
    emb = model.encode(texts, convert_to_numpy=True, batch_size=batch_size).astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)  # normalize -> cosine via inner product
    return emb

def build_jobs_index(
    input_folder: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    faiss_index_path: str = "../data/job_treated/jobs_index.faiss",
    mapping_path: str = "../data/job_treated/jobs_index_mapping.json",
):

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Folder not found: {input_folder}")

    files, texts = _read_txt_files(input_folder)
    if not files:
        raise FileNotFoundError(f"No .txt job files found in: {input_folder}")

    model = SentenceTransformer(model_name)

    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding jobs"):
        batch = texts[start:start + batch_size]
        vectors.append(_encode_texts(model, batch, batch_size=batch_size))
    vectors = np.vstack(vectors)

    # Save mapping: faiss position -> filename
    os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False, indent=2)

    # Build FAISS index
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(os.path.dirname(faiss_index_path) or ".", exist_ok=True)
    faiss.write_index(index, faiss_index_path)

    return {
        "count": len(files),
        "dim": int(vectors.shape[1]),
        "faiss_index_path": faiss_index_path,
        "mapping_path": mapping_path
    }
