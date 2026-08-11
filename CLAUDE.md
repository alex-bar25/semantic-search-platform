# Project rules

## What this is
A small semantic search platform for a CV portfolio: hybrid keyword + vector
retrieval, with a PyTorch-trained cross-encoder re-ranker on top, served via
FastAPI. The point is to demonstrate PyTorch, RAG, vector DB, and Pydantic
skills clearly and simply. It is not a product. It is not infrastructure.

## Context: keep this manageable
I have ADHD. Long files, deep abstractions, and sprawling multi-file changes
make this project hard for me to hold in my head and finish. Every rule below
exists to protect that. When in doubt, choose the smaller, more boring option.

## Hard rules
- Prefer fewer files over more files. If a new file feels tempting, ask
  first whether it can just be a function in an existing one.
- No premature abstraction. No base classes, plugin systems, or config
  frameworks for a project this size. Write the direct thing.
- Cap each file at roughly 150-200 lines. If it's getting longer, stop and
  say so rather than continuing to add to it.
- One change at a time. Don't restructure multiple files in a single pass.
  Propose the change, do it, confirm it runs, then move on.
- No new dependencies outside the stack list below without asking first.
- No Docker, no CI/CD, no auth, no frontend, no distributed anything. If a
  task starts pulling in that direction, flag it instead of building it.
- Simple code over clever code. Explicit over implicit. A slightly
  repetitive but readable function beats a clever one-liner.

## Stack (do not expand without asking)
- PyTorch (MPS backend) for the cross-encoder re-ranker
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- Chroma for the vector index
- SQLite FTS5 for the keyword index
- FastAPI + Pydantic for the API layer
- pytest for tests
- Weights & Biases (free tier) for training logs, optional
- LangChain only if it genuinely simplifies something; otherwise skip it
- UV

## Structure (target, don't exceed without discussion)
```
data/          corpus + eval query set
index/         keyword + vector indexing code
rerank/        training script, model, inference
api/           FastAPI app + Pydantic models
eval/          nDCG/MRR scoring scripts
tests/
README.md      what it is, why each layer exists, results table
```

## When starting a task
Before writing code, say in one or two sentences what you're about to build
and which file(s) it touches. If it's more than that, break it down and
confirm the first small step before continuing.

## Definition of done for this project
- Hybrid retrieval works end to end
- Re-ranker is trained and measurably improves nDCG@10 or MRR over baseline
- README has a results table and a short "why" for each architectural choice
- Nothing beyond that is required. Resist adding more.