# Movie RAG Search Engine

> **Work in Progress** — Building toward a full Retrieval-Augmented Generation (RAG) search engine over a movie database, one search strategy at a time.

---

## Overview

This project is an exploration of information retrieval techniques applied to a movie dataset, progressing from simple baseline methods toward a full RAG pipeline powered by vector embeddings and a large language model.

The goal is to understand how each retrieval strategy works, where it falls short, and what motivates the next step in the stack — culminating in a semantic search + LLM answer synthesis system.

---

## Current Status

| Stage | Method | Status |
|-------|--------|--------|
| 1 | Keyword Search | ✅ Implemented |
| 2 | BM25 Search | ✅ Implemented |
| 3 | Dense Vector / Embedding Search | 🔜 Planned |
| 4 | RAG (Retrieval + Generation) | 🔜 Planned |

---

## Search Methods

### 1. Keyword Search

The simplest possible baseline: scan each document for exact string matches against the query.

**How it works:**
- Tokenize the query into individual terms
- Return documents where any (or all) terms appear in the target fields (title, description, genre, etc.)
- Rank by match count or use a basic TF heuristic

**What we learned:**
- Fast and completely interpretable — easy to debug why a result was or wasn't returned
- Brittle against synonyms, plurals, and alternate phrasings (searching "spaceship" won't find "spacecraft")
- No sense of term importance — the word "the" is treated the same as "Spielberg"
- Ranking quality is poor; results are often arbitrary among the matched set
- Establishes a useful lower-bound baseline to measure future improvements against

---

### 2. BM25 Search

BM25 (Best Match 25) is a probabilistic ranking function that improves significantly on raw keyword matching. It's the industry-standard baseline for lexical search, used under the hood by Elasticsearch and many production systems.

**How it works:**
- Scores each document based on term frequency (TF) — but with diminishing returns for repeated terms
- Applies inverse document frequency (IDF) — rare terms that appear in fewer documents are weighted more heavily
- Normalizes for document length — a term appearing in a short document is more significant than in a long one
- Final score: `score(D, Q) = Σ IDF(qᵢ) · [TF(qᵢ, D) · (k1 + 1)] / [TF(qᵢ, D) + k1 · (1 - b + b · |D|/avgdl)]`

**What we learned:**
- Dramatically better ranking than keyword search — documents most "about" the query rise to the top
- IDF naturally suppresses noise from common words without needing a stopword list
- Document length normalization prevents long movie descriptions from dominating results unfairly
- Still entirely lexical — it cannot handle semantic similarity. "Heist film" and "robbery movie" are invisible to each other
- Performs surprisingly well on precise factual queries (director names, actor names, exact titles)
- Falls apart on vague or conceptual queries ("something scary but not gory", "movies like Interstellar")
- Makes clear why we need embeddings: the vocabulary mismatch problem is fundamental to any token-matching approach

---

## Roadmap

The limitations identified above directly motivate the next phases:

**Phase 3 — Dense Embedding Search**
Encode movies and queries as high-dimensional vectors using a sentence transformer model. Semantic similarity replaces token overlap, so "heist" and "robbery" are close neighbors in embedding space.

**Phase 4 — Full RAG Pipeline**
Retrieve the top-k relevant movies via embedding search, then pass them as context to an LLM. The model synthesizes a natural-language answer grounded in the actual database content — enabling queries like "What's a good sci-fi film about isolation for a Friday night?"

---

## Project Structure

```
RAG/
├── cli/                 
|     ├── lib/
|     |    ├── keyword_search.py    # InvertedIndex class, keyword search, and BM25 search functions
|     |    └── search_utils.py      # General utilities needed for search functions
|     ├── keyword_search_cli.py     # Command-line interface for running searches
```

---

## Why This Approach?

Building retrieval methods from the ground up — rather than jumping straight to a vector database — makes it possible to understand exactly what each abstraction buys you. Every limitation discovered in keyword search motivates BM25. Every limitation of BM25 motivates embeddings. Every limitation of retrieval alone motivates generation.

This project is a learning-first implementation.

---

## Tech Stack

- **Python** — core implementation language
- **CLI** — custom command-line interface for running and comparing searches
- *Embeddings, vector store, and LLM integration — coming in future phases*

---

## Contributing

This is a personal learning project, but feedback and suggestions are welcome via Issues.
