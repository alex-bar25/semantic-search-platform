from data.load import corpus_from_rows, qrels_from_rows, queries_from_rows


def test_corpus_joins_title_and_text():
    rows = [{"_id": "d1", "title": "ETFs explained", "text": "An ETF is a fund."}]
    assert corpus_from_rows(rows) == {"d1": "ETFs explained. An ETF is a fund."}


def test_corpus_handles_empty_title():
    rows = [{"_id": "d2", "title": "", "text": "Body only."}]
    assert corpus_from_rows(rows) == {"d2": "Body only."}


def test_queries_from_rows():
    rows = [{"_id": "q1", "text": "what is an etf"}]
    assert queries_from_rows(rows) == {"q1": "what is an etf"}


def test_qrels_groups_by_query_and_drops_zero_scores():
    rows = [
        {"query-id": "q1", "corpus-id": "d1", "score": 1},
        {"query-id": "q1", "corpus-id": "d2", "score": 2},
        {"query-id": "q2", "corpus-id": "d3", "score": 1},
        {"query-id": "q2", "corpus-id": "d4", "score": 0},
    ]
    assert qrels_from_rows(rows) == {
        "q1": {"d1": 1, "d2": 2},
        "q2": {"d3": 1},
    }


def test_qrels_coerces_integer_ids_to_strings():
    rows = [{"query-id": 8, "corpus-id": 566392, "score": 1}]
    assert qrels_from_rows(rows) == {"8": {"566392": 1}}
