import os
import re
import sqlite3

DB_PATH = "data/keyword.db"


def build(corpus: dict[str, str], db_path: str = DB_PATH) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, text)")
    conn.executemany("INSERT INTO docs (doc_id, text) VALUES (?, ?)", corpus.items())
    conn.commit()
    conn.close()


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"No keyword index at {db_path}. Build it with: uv run python -m index.keyword"
        )
    return sqlite3.connect(db_path, check_same_thread=False)


def _to_match_query(query: str) -> str:
    tokens = re.findall(r"\w+", query)
    return " OR ".join(f'"{token}"' for token in tokens)


def search(
    conn: sqlite3.Connection, query: str, k: int = 100
) -> list[tuple[str, float]]:
    match_query = _to_match_query(query)
    if not match_query:
        return []
    rows = conn.execute(
        "SELECT doc_id, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
        (match_query, k),
    ).fetchall()
    return [(doc_id, -score) for doc_id, score in rows]


if __name__ == "__main__":
    from data.load import load_corpus

    corpus = load_corpus()
    build(corpus)
    print(f"Indexed {len(corpus)} documents into {DB_PATH}")
