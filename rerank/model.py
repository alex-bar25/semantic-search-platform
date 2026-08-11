import os
from pathlib import Path

from sentence_transformers import CrossEncoder

BACKBONE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHECKPOINT_PATH = str(Path(__file__).resolve().parent / "checkpoint")
MAX_LENGTH = 256


def new_model() -> CrossEncoder:
    return CrossEncoder(BACKBONE, num_labels=1, max_length=MAX_LENGTH)


def load_trained(path: str = CHECKPOINT_PATH) -> CrossEncoder:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained re-ranker at {path}. Train one with: uv run python -m rerank.train"
        )
    return CrossEncoder(path, num_labels=1, max_length=MAX_LENGTH)


def rerank(
    model: CrossEncoder, query: str, doc_ids: list[str], texts: list[str]
) -> list[tuple[str, float]]:
    if not doc_ids:
        return []
    scores = model.predict([(query, text) for text in texts])
    ranked = sorted(zip(doc_ids, scores), key=lambda pair: -pair[1])
    return [(doc_id, float(score)) for doc_id, score in ranked]
