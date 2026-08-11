from collections.abc import Iterable

from datasets import load_dataset

CORPUS_REPO = "BeIR/fiqa"
QRELS_REPO = "BeIR/fiqa-qrels"


def corpus_from_rows(rows: Iterable[dict]) -> dict[str, str]:
    corpus = {}
    for row in rows:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        corpus[row["_id"]] = f"{title}. {text}" if title else text
    return corpus


def qrels_from_rows(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        score = int(row["score"])
        if score <= 0:
            continue
        query_id = str(row["query-id"])
        qrels.setdefault(query_id, {})[str(row["corpus-id"])] = score
    return qrels


def load_corpus() -> dict[str, str]:
    return corpus_from_rows(load_dataset(CORPUS_REPO, "corpus")["corpus"])


def load_queries() -> dict[str, str]:
    rows = load_dataset(CORPUS_REPO, "queries")["queries"]
    return {row["_id"]: row["text"] for row in rows}


def load_qrels(split: str) -> dict[str, dict[str, int]]:
    return qrels_from_rows(load_dataset(QRELS_REPO)[split])


if __name__ == "__main__":
    corpus = load_corpus()
    queries = load_queries()
    test_qrels = load_qrels("test")
    print(f"corpus: {len(corpus)} docs")
    print(f"queries: {len(queries)}")
    print(f"test qrels: {len(test_qrels)} queries with judgements")
