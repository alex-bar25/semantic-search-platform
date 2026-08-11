import pytrec_eval


def evaluate(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    scored_run = {
        query_id: {
            doc_id: float(k - rank) for rank, doc_id in enumerate(ranked_ids[:k])
        }
        for query_id, ranked_ids in run.items()
        if query_id in qrels and ranked_ids
    }
    if not scored_run:
        return {f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0, "queries": 0}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}", "recip_rank"})
    per_query = evaluator.evaluate(scored_run)

    ndcgs = [m[f"ndcg_cut_{k}"] for m in per_query.values()]
    mrrs = [m["recip_rank"] for m in per_query.values()]
    return {
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        f"mrr@{k}": sum(mrrs) / len(mrrs),
        "queries": len(ndcgs),
    }
