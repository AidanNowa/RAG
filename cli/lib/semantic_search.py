import os

import numpy as np
import re

from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, load_movies

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list(dict())):
        self.documents = documents
        document_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            document_list.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(document_list, show_progress_bar=True)
        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list(dict())):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
    
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        self.build_embeddings(documents)
        return self.embeddings

    def search(self, query, limit):
        if len(self.embeddings) == 0 or self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if len(self.documents) == 0 or self.documents is None:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")    
    
        query_embedding = self.generate_embedding(query)

        sim_scores = []
        for i, doc_embedding  in enumerate(self.embeddings):
            cosine_sim = cosine_similarity(query_embedding, doc_embedding)
            sim_scores.append((cosine_sim, self.documents[i]))

        sorted_scores = sorted(sim_scores, key=lambda x: x[0], reverse=True)

        result = []
        for i in range(limit):
            result.append(
                {
                    'score': sorted_scores[i][0], 
                    'title': sorted_scores[i][1]['title'], 
                    'description': sorted_scores[i][1]['description']
                }
            )

        return result

def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text: str):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {search_instance.embeddings.shape[0]} vectors in {search_instance.embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def search_command(query, limit=5):
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    results = search_instance.search(query, limit)
    
    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} ({result['score']})\n   {result['description'][:100]}...\n\n")

def fixed_size_chunking(text: str, chunk_size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []

    word_count = len(words)
    i = 0
    while i < word_count:
        chunk_words = words[i:i+chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks

def chunk_text(text: str, chunk_size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunking(text: str, max_chunk_size: int=4, overlap: int=DEFAULT_CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    sentence_count = len(sentences)
    i = 0
    while i < sentence_count:
        chunk_sentences = sentences[i:i+max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ". join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


