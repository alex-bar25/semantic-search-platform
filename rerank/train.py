from sentence_transformers import InputExample
from torch.utils.data import DataLoader

from rerank.model import CHECKPOINT_PATH, new_model

NEGATIVES_PER_POSITIVE = 7
MINE_K = 50
EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5


def select_negatives(candidates: list[str], positives: set[str], n: int) -> list[str]:
    negatives = [doc_id for doc_id in candidates if doc_id not in positives]
    return negatives[:n]


def build_examples(
    engine,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, str],
    negatives_per_positive: int = NEGATIVES_PER_POSITIVE,
    limit: int | None = None,
) -> list[InputExample]:
    examples = []
    query_ids = list(qrels)[:limit] if limit else list(qrels)
    for i, query_id in enumerate(query_ids):
        query_text = queries.get(query_id)
        positives = set(qrels[query_id])
        if not query_text or not positives:
            continue
        candidates = [d for d, _ in engine.search(query_text, "hybrid", MINE_K)]
        negatives = select_negatives(candidates, positives, negatives_per_positive)
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
    from data.load import load_corpus, load_qrels, load_queries
    from search.engine import SearchEngine

    engine = SearchEngine()
    corpus = load_corpus()
    queries = load_queries()
    train_qrels = load_qrels("train")

    print(f"Mining hard negatives over {len(train_qrels)} training queries...")
    examples = build_examples(engine, queries, train_qrels, corpus)
    positives = sum(1 for e in examples if e.label == 1.0)
    print(f"Built {len(examples)} examples ({positives} positive)")

    train(examples)
