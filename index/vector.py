from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from data.load import load_corpus

CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma")
COLLECTION_NAME = "fiqa"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256
ADD_CHUNK = 5000


def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def build(corpus: dict[str, str], path: str = CHROMA_PATH) -> None:
    client = chromadb.PersistentClient(path=path)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    model = get_model()
    doc_ids = list(corpus.keys())
    texts = [corpus[doc_id] for doc_id in doc_ids]
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    )

    for start in range(0, len(doc_ids), ADD_CHUNK):
        stop = start + ADD_CHUNK
        collection.add(
            ids=doc_ids[start:stop],
            embeddings=embeddings[start:stop].tolist(),
            documents=texts[start:stop],
        )


def connect(path: str = CHROMA_PATH):
    client = chromadb.PersistentClient(path=path)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise FileNotFoundError(
            f"No vector index at {path}. Build it with: uv run python -m index.vector"
        ) from exc


def search(collection, model, query: str, k: int = 100) -> list[tuple[str, float]]:
    embedding = model.encode([query], convert_to_numpy=True)
    result = collection.query(query_embeddings=embedding.tolist(), n_results=k)
    doc_ids = result["ids"][0]
    distances = result["distances"][0]
    return [(doc_id, 1.0 - distance) for doc_id, distance in zip(doc_ids, distances)]


if __name__ == "__main__":
    corpus = load_corpus()
    build(corpus)
    print(f"Embedded {len(corpus)} documents into {CHROMA_PATH}")
