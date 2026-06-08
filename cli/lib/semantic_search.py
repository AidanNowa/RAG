import os

import numpy as np
import re
import json

from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_SEMANTIC_CHUNK_SIZE,  DEFAULT_CHUNK_OVERLAP, load_movies, format_search_result

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
MOVIE_CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
MOVIE_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        metadata = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue
            
            doc_chunks = semantic_chunking(text, max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
            
            for i, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                metadata.append(
                    {
                        "movie_idx": idx, 
                        "chunk_idx": i, 
                        "total_chunks": len(doc_chunks)
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        os.makedirs(os.path.dirname(MOVIE_CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        
        with open(MOVIE_METADATA_PATH, 'w') as f:
            json.dump({"chunks": metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        
        return self.chunk_embeddings


    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        if os.path.exists(MOVIE_CHUNK_EMBEDDINGS_PATH) and os.path.exists(MOVIE_METADATA_PATH):
            self.chunk_embeddings = np.load(MOVIE_CHUNK_EMBEDDINGS_PATH)
            with open(MOVIE_METADATA_PATH, 'r') as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)


    def search_chunks(self, query: str, limit: int=10):
        query_embedding = self.generate_embedding(query)
        chunk_score = [dict()]
        
        curr_movie_idx = None
        chunk_idx = None
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            if self.chunk_metadata[i]["movie_idx"] != curr_movie_idx:
                chunk_idx = 0
                curr_movie_idx = self.chunk_metadata[i]["movie_idx"]
            cosine_sim = cosine_similarity(query_embedding, chunk_embedding)
            chunk_score.append(
                {
                    "chunk_idx": chunk_idx,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": cosine_sim
                }
            )           
            chunk_idx += 1
        movie_score_map = {} 
        for i in range(1, len(chunk_score)):
            score = chunk_score[i]
            if score["movie_idx"] not in movie_score_map or score["score"] > movie_score_map[score["movie_idx"]]:
                movie_score_map[score["movie_idx"]] = score["score"]
        
        sorted_scores = dict(sorted(movie_score_map.items(), key=lambda item: item[1], reverse=True))

        results = []
        for i in range(limit):
            movie_idx = list(sorted_scores.keys())[i]
            doc = self.documents[movie_idx]
            results.append(format_search_result(movie_idx, doc["title"], doc["description"][:100], sorted_scores[movie_idx]))        
        return results        

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
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith(('.', '!', '?')):
            sentences = [text]

    chunks = []
    sentence_count = len(sentences)
    i = 0

    while i < sentence_count:
        chunk_sentences = sentences[i:i+max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        stripped_chunk_sentences = []
        for sentence in chunk_sentences:
            sentence = sentence.strip()
            if sentence:
                stripped_chunk_sentence.append(sentence)
        if not stripped_chunk_sentences:
            i += max_chunk_size - overlap
            continue
            
        chunks.append(" ". join(stripped_chunk_sentences))
        i += max_chunk_size - overlap
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def embed_chunks_command() -> None:
    search_instance = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = search_instance.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(text: str, limit: int=5) -> None:
    search_instance = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = search_instance.load_or_create_chunk_embeddings(movies)
    results = search_instance.search_chunks(text, limit)
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result["title"]} (score: {result["score"]:.4f})")
        print(f"    {result["document"]}...")
    
