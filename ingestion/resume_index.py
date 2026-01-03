import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def _read_txt_files(input_folder: str):
    # Listing and reading .txt resumes
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".txt")])
    paths = [os.path.join(input_folder, f) for f in files]

    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return files, texts


def _encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int):
    # Encoding texts into normalized vectors (cosine-ready)
    emb = model.encode(texts, convert_to_numpy=True, batch_size=batch_size).astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)
    return emb


def build_resume_index(
    input_folder: str,
    backend: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    # FAISS outputs
    faiss_index_path: str = "../data/resume_index.faiss",
    mapping_path: str = "../data/resume_index_mapping.json",
    # Chroma outputs
    chroma_dir: str = "../data/chroma_resume_db",
    chroma_collection: str = "resumes",
):
    # Building a resume index using FAISS or Chroma
    backend = backend.lower().strip()
    if backend not in {"faiss", "chroma"}:
        raise ValueError("backend must be 'faiss' or 'chroma'")

    model = SentenceTransformer(model_name)
    files, texts = _read_txt_files(input_folder)

    if not files:
        raise FileNotFoundError(f"No .txt files found in: {input_folder}")

    # Encoding all resumes
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding resumes"):
        batch = texts[start : start + batch_size]
        vectors.append(_encode_texts(model, batch, batch_size=batch_size))

    vectors = np.vstack(vectors)

    # Saving mapping (useful for FAISS or debugging)
    os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False, indent=2)

    if backend == "faiss":
        # Building FAISS index (inner product on normalized vectors = cosine similarity)
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError(
                "FAISS is not available. Install faiss-cpu or use backend='chroma'."
            ) from e

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        os.makedirs(os.path.dirname(faiss_index_path) or ".", exist_ok=True)
        faiss.write_index(index, faiss_index_path)

        return {
            "backend": "faiss",
            "count": len(files),
            "dim": int(vectors.shape[1]),
            "faiss_index_path": faiss_index_path,
            "mapping_path": mapping_path,
        }

    # backend == "chroma"
    import chromadb

    # Creating persistent Chroma client
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)

    # Recreating collection cleanly (simple approach)
    try:
        client.delete_collection(chroma_collection)
    except Exception:
        pass
    col = client.create_collection(name=chroma_collection)

    # Adding items with metadata (filename)
    ids = [f"resume_{i}" for i in range(len(files))]
    metadatas = [{"filename": files[i]} for i in range(len(files))]
    documents = texts  # storing the semantic text

    col.add(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return {
        "backend": "chroma",
        "count": len(files),
        "dim": int(vectors.shape[1]),
        "chroma_dir": chroma_dir,
        "collection": chroma_collection,
        "mapping_path": mapping_path,
    }
