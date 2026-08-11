from sentence_transformers import InputExample

from rerank.train import build_examples, load_examples, save_examples, select_negatives


def test_examples_survive_a_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "examples.json")
    original = [
        InputExample(texts=["q", "positive"], label=1.0),
        InputExample(texts=["q", "negative"], label=0.0),
    ]
    save_examples(original, path)
    restored = load_examples(path)
    assert [(e.texts, e.label) for e in restored] == [
        (["q", "positive"], 1.0),
        (["q", "negative"], 0.0),
    ]


def test_select_negatives_excludes_positives():
    negatives = select_negatives(["d1", "d2", "d3", "d4"], positives={"d2"}, n=2)
    assert "d2" not in negatives
    assert len(negatives) == 2


def test_select_negatives_preserves_retriever_order():
    assert select_negatives(["d1", "d2", "d3"], positives=set(), n=2) == ["d1", "d2"]


def test_select_negatives_returns_fewer_when_not_enough_candidates():
    assert select_negatives(["d1"], positives={"d1"}, n=3) == []


class FakeEngine:
    def search(self, query, mode="hybrid", top_k=10):
        return [("n1", 1.0), ("p1", 0.9), ("n2", 0.8), ("n3", 0.7)]


CORPUS = {
    "p1": "positive passage",
    "n1": "hard negative one",
    "n2": "hard negative two",
    "n3": "hard negative three",
}


def test_build_examples_labels_positives_and_negatives():
    examples = build_examples(
        FakeEngine(),
        {"q1": "a query"},
        {"q1": {"p1": 1}},
        CORPUS,
        negatives_per_query=2,
    )
    labels = sorted(e.label for e in examples)
    assert labels == [0.0, 0.0, 1.0]


def test_build_examples_excludes_the_positive_from_negatives():
    examples = build_examples(
        FakeEngine(),
        {"q1": "a query"},
        {"q1": {"p1": 1}},
        CORPUS,
        negatives_per_query=3,
    )
    negative_texts = [e.texts[1] for e in examples if e.label == 0.0]
    assert CORPUS["p1"] not in negative_texts


def test_build_examples_pairs_each_text_with_the_query():
    examples = build_examples(
        FakeEngine(), {"q1": "a query"}, {"q1": {"p1": 1}}, CORPUS,
        negatives_per_query=1,
    )
    assert all(e.texts[0] == "a query" for e in examples)


def test_build_examples_skips_queries_missing_from_the_queries_file():
    examples = build_examples(
        FakeEngine(), {}, {"q1": {"p1": 1}}, CORPUS, negatives_per_query=1
    )
    assert examples == []


def test_build_examples_skips_docs_missing_from_the_corpus():
    examples = build_examples(
        FakeEngine(),
        {"q1": "a query"},
        {"q1": {"missing_doc": 1}},
        CORPUS,
        negatives_per_query=1,
    )
    assert all(e.label == 0.0 for e in examples)


def test_build_examples_respects_limit():
    queries = {"q1": "one", "q2": "two", "q3": "three"}
    qrels = {"q1": {"p1": 1}, "q2": {"p1": 1}, "q3": {"p1": 1}}
    examples = build_examples(
        FakeEngine(), queries, qrels, CORPUS, negatives_per_query=1, limit=2
    )
    assert len({e.texts[0] for e in examples}) == 2
