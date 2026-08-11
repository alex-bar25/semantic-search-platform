import pytest

from index import keyword

FIXTURE = {
    "d1": "An ETF is an exchange traded fund holding many assets.",
    "d2": "A mutual fund pools money from many investors.",
    "d3": "Compound interest grows savings over time.",
}


@pytest.fixture
def conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    keyword.build(FIXTURE, db_path)
    connection = keyword.connect(db_path)
    yield connection
    connection.close()


def test_search_finds_exact_term(conn):
    results = keyword.search(conn, "ETF", k=10)
    assert results[0][0] == "d1"


def test_search_returns_descending_scores(conn):
    results = keyword.search(conn, "fund", k=10)
    assert len(results) == 2
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(conn):
    assert len(keyword.search(conn, "fund", k=1)) == 1


def test_search_survives_punctuation(conn):
    results = keyword.search(conn, "What's an ETF? (really)", k=10)
    assert results[0][0] == "d1"


def test_search_with_no_searchable_tokens_returns_empty(conn):
    assert keyword.search(conn, "???", k=10) == []


def test_search_with_no_matches_returns_empty(conn):
    assert keyword.search(conn, "cryptocurrency", k=10) == []


def test_connect_without_index_raises():
    with pytest.raises(FileNotFoundError):
        keyword.connect("/nonexistent/path/keyword.db")


def test_build_replaces_existing_index(tmp_path):
    db_path = str(tmp_path / "rebuild.db")
    keyword.build(FIXTURE, db_path)
    keyword.build({"d9": "An ETF only."}, db_path)
    connection = keyword.connect(db_path)
    results = keyword.search(connection, "ETF", k=10)
    connection.close()
    assert [doc_id for doc_id, _ in results] == ["d9"]
