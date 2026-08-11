import json
import os
from pathlib import Path

from sentence_transformers import InputExample
from torch.utils.data import DataLoader

from data.load import load_corpus, load_qrels, load_queries
from rerank.model import CHECKPOINT_PATH, new_model
from search.engine import RERANK_K, SearchEngine

NEGATIVES_PER_QUERY = 7
EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EXAMPLES_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "train_examples.json"
)


def select_negatives(candidates: list[str], positives: set[str], n: int) -> list[str]:
    negatives = [doc_id for doc_id in candidates if doc_id not in positives]
    return negatives[:n]


def build_examples(
    engine,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, str],
    negatives_per_query: int = NEGATIVES_PER_QUERY,
    limit: int | None = None,
) -> list[InputExample]:
    examples = []
    query_ids = list(qrels)[:limit] if limit else list(qrels)
    for i, query_id in enumerate(query_ids):
        query_text = queries.get(query_id)
        positives = set(qrels[query_id])
        if not query_text or not positives:
            continue
        candidates = [d for d, _ in engine.search(query_text, "hybrid", RERANK_K)]
        negatives = select_negatives(candidates, positives, negatives_per_query)
        for positive_id in positives:
            if positive_id in corpus:
                examples.append(
                    InputExample(texts=[query_text, corpus[positive_id]], label=1.0)
                )
        for negative_id in negatives:
            if negative_id in corpus:
                examples.append(
                    InputExample(texts=[query_text, corpus[negative_id]], label=0.0)
                )
        if (i + 1) % 250 == 0:
            print(f"mined {i + 1}/{len(query_ids)} queries, {len(examples)} examples")
    return examples


def save_examples(examples, path: str = EXAMPLES_PATH) -> None:
    rows = [{"texts": e.texts, "label": e.label} for e in examples]
    with open(path, "w") as handle:
        json.dump(rows, handle)


def load_examples(path: str = EXAMPLES_PATH) -> list[InputExample]:
    with open(path) as handle:
        rows = json.load(handle)
    return [InputExample(texts=r["texts"], label=r["label"]) for r in rows]


def train(examples, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    model = new_model()
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=int(0.1 * len(loader) * epochs),
        optimizer_params={"lr": LEARNING_RATE},
        output_path=CHECKPOINT_PATH,
        show_progress_bar=True,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    return model


if __name__ == "__main__":
    if os.path.exists(EXAMPLES_PATH):
        examples = load_examples()
        print(f"Reusing {len(examples)} mined examples from {EXAMPLES_PATH}")
    else:
        engine = SearchEngine()
        train_qrels = load_qrels("train")
        print(f"Mining hard negatives over {len(train_qrels)} training queries...")
        examples = build_examples(
            engine, load_queries(), train_qrels, load_corpus()
        )
        save_examples(examples)
        positives = sum(1 for e in examples if e.label == 1.0)
        print(f"Built {len(examples)} examples ({positives} positive)")

    train(examples)
