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
| 1 | Keyword Search | Implemented |
| 2 | BM25 Search | Implemented |
| 3 | Semantic Search | Implemented |
| 4 | Hybrid Search | Planned |
| 5 | RAG (Retrieval + Generation) | Planned |

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
- Scores each document based on term frequency (TF) (but with diminishing returns for repeated terms)
- Applies inverse document frequency (IDF): rare terms that appear in fewer documents are weighted more heavily
- Normalizes for document length: a term appearing in a short document is more significant than in a long one
- Final score: `score(D, Q) = Σ IDF(qᵢ) · [TF(qᵢ, D) · (k1 + 1)] / [TF(qᵢ, D) + k1 · (1 - b + b · |D|/avgdl)]`

**What we learned:**
- Better ranking than keyword search, documents most "about" the query rise to the top
- IDF naturally suppresses noise from common words without needing a stopword list
- Document length normalization prevents long movie descriptions from dominating results unfairly
- Still entirely lexical, it cannot handle semantic similarity. "Heist film" and "robbery movie" are invisible to each other
- Performs well on precise factual queries (director names, actor names, exact titles)
- Falls apart on vague or conceptual queries ("something scary but not gory", "movies like Interstellar")
- Makes clear why we need embeddings: the vocabulary mismatch problem is fundamental to any token-matching approach

---

### 3. Semantic Search (Dense Embeddings)
 
Semantic search encodes both documents and queries as dense vectors in a shared high-dimensional space, so that meaning determines relevance. Two texts with completely different words can be near neighbors if they convey the same idea.
 
**How it works:**
- Each movie's text (title, description, genre, etc.) is encoded into a fixed-size embedding vector using a sentence transformer model
- At query time, the query is embedded using the same model
- Cosine similarity between the query vector and every document vector determines ranking
- Supports both whole-document embeddings and chunked embeddings, where long descriptions are split into overlapping segments before encoding

**Chunking strategies supported:**
- Fixed-size chunking: splits text into chunks of N words with optional word overlap between chunks (`chunk`, `embed_chunks`)
- Semantic chunking: groups sentences into coherent chunks by meaning boundary rather than arbitrary word count, with optional sentence overlap (`semantic_chunk`)
- Chunked search (`search_chunked`): retrieves at the chunk level, then deduplicates back to movie level

**What we learned:**
 
*Where semantic search wins over BM25:*
- Understands synonyms and paraphrases naturally: searching `"bear attack"` surfaces *The Revenant* ("mauled by grizzly") and *Grizzly Man* ("dangerous bear interactions"), not just films with those exact words
- Handles conceptual queries that have no lexical anchor: "uplifting", "feel-good", "something scary but not gory" can yield meaningful results
- Context disambiguation works implicitly: "bank" in a crime thriller query doesn't confuse it with river documentaries
- Vague, open-ended queries ("movies like Interstellar") are where semantic search is useful
*Where semantic search still falls short:*
- Exact string precision is weaker: `"Toy Story 3"` and `"Toy Story"` may score similarly because the embedding captures the general concept of the franchise
- Rare proper nouns, model numbers, or very new slang may be poorly represented in the embedding model's training distribution
- Results are harder to interpret and debug: there's no clear explanation for why one result ranked above another
- Computationally heavier than BM25: embedding the full corpus is an upfront cost, and similarity search scales with corpus size
*On chunking:*
- Whole-document embeddings can dilute a specific detail buried in a long description; chunking lets precise passages surface
- Semantic chunking tends to produce more coherent retrieval units than fixed-size, but is more expensive to compute
- Overlap between chunks reduces the risk of a key sentence being split across a boundary and losing context

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
