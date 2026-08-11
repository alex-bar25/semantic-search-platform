import pytrec_eval

MODES = ("keyword", "vector", "hybrid", "rerank")


def evaluate(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    judged = {
        query_id: ranked_ids
        for query_id, ranked_ids in run.items()
        if query_id in qrels
    }
    if not judged:
        return {f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0, "queries": 0}

    scored_run = {
        query_id: {
            doc_id: float(k - rank) for rank, doc_id in enumerate(ranked_ids[:k])
        }
        for query_id, ranked_ids in judged.items()
        if ranked_ids
    }

    per_query = {}
    if scored_run:
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {f"ndcg_cut.{k}", "recip_rank"}
        )
        per_query = evaluator.evaluate(scored_run)

    ndcgs = [per_query.get(q, {}).get(f"ndcg_cut_{k}", 0.0) for q in judged]
    mrrs = [per_query.get(q, {}).get("recip_rank", 0.0) for q in judged]
    return {
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        f"mrr@{k}": sum(mrrs) / len(mrrs),
        "queries": len(ndcgs),
    }


def evaluate_mode(engine, queries, qrels, mode: str, k: int = 10) -> dict[str, float]:
    run = {
        query_id: [doc_id for doc_id, _ in engine.search(queries[query_id], mode, k)]
        for query_id in qrels
    }
    return evaluate(run, qrels, k)


if __name__ == "__main__":
    import os

    from data.load import load_qrels, load_queries
    from rerank.model import CHECKPOINT_PATH
    from search.engine import SearchEngine

    queries = load_queries()
    qrels = load_qrels("test")
    trained = os.path.exists(CHECKPOINT_PATH)
    engine = SearchEngine(load_reranker=trained)

    print(f"\nFiQA-2018 test set, {len(qrels)} queries\n")
    print("| config | nDCG@10 | MRR@10 |")
    print("|---|---|---|")
    for mode in MODES:
        if mode == "rerank" and not trained:
            print("| rerank | not trained yet | not trained yet |")
            continue
        result = evaluate_mode(engine, queries, qrels, mode)
        print(f"| {mode} | {result['ndcg@10']:.4f} | {result['mrr@10']:.4f} |")
