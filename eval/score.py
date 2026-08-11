import math


def dcg(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    gains = [relevant.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal_gains = sorted(relevant.values(), reverse=True)
    ideal = dcg(ideal_gains, k)
    if ideal == 0.0:
        return 0.0
    return dcg(gains, k) / ideal


def mrr_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    for i, doc_id in enumerate(ranked_ids[:k]):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    ndcgs = []
    mrrs = []
    for query_id, ranked_ids in run.items():
        relevant = qrels.get(query_id)
        if not relevant:
            continue
        ndcgs.append(ndcg_at_k(ranked_ids, relevant, k))
        mrrs.append(mrr_at_k(ranked_ids, relevant, k))
    if not ndcgs:
        return {f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0, "queries": 0}
    return {
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        f"mrr@{k}": sum(mrrs) / len(mrrs),
        "queries": len(ndcgs),
    }
