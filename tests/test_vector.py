import pytest

from index import vector

FIXTURE = {
    "d1": "An ETF is an exchange traded fund holding many assets.",
    "d2": "A mutual fund pools money from many investors.",
    "d3": "Compound interest grows savings over time.",
}


@pytest.fixture(scope="module")
def model():
    return vector.get_model()


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("chroma"))
    vector.build(FIXTURE, path)
    return vector.connect(path)


def test_pick_device_returns_known_device():
    assert vector.pick_device() in {"mps", "cpu"}


def test_search_finds_semantically_related_doc(collection, model):
    results = vector.search(collection, model, "how does saving money grow", k=3)
    assert results[0][0] == "d3"


def test_search_returns_descending_scores(collection, model):
    results = vector.search(collection, model, "investment fund", k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(collection, model):
    assert len(vector.search(collection, model, "fund", k=2)) == 2


def test_connect_without_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vector.connect(str(tmp_path / "missing"))


def test_build_replaces_existing_collection(tmp_path, model):
    path = str(tmp_path / "rebuild")
    vector.build(FIXTURE, path)
    vector.build({"d9": "An ETF only."}, path)
    collection = vector.connect(path)
    results = vector.search(collection, model, "ETF", k=10)
    assert [doc_id for doc_id, _ in results] == ["d9"]
