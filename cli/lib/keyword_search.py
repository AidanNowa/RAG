import string
import pickle
import os
import math
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from .search_utils import DEFAULT_SEARCH_LIMIT, CACHE_DIR, load_movies, load_stopwords


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
 
    def __add_document(self, doc_id, text) -> None:
        tokens = clean_tokens(text)
        for token in tokens:
            if doc_id not in self.index[token]:
                self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1 
    
    def get_documents(self, term) -> list[int]:
        return list(self.index.get(term.lower(), set())).sort()

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def save(self) -> None:
        #print(f"{self.index['merida']}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as index_file:
            pickle.dump(self.index, index_file)
        with open(self.docmap_path, 'wb') as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open(self.term_frequencies_path, 'wb') as term_frequencies_file:
            pickle.dump(self.term_frequencies, term_frequencies_file)

    def load(self):
        try:
            with open(self.index_path, "rb") as index_file:
                index = pickle.load(index_file)
                self.index = index
        except FileNotFoundError:
            print(f"Error: the file '{self.index_path}' does not exist.")
        try:
            with open(self.docmap_path, "rb") as docmap_file:
                docmap = pickle.load(docmap_file)
                self.docmap = docmap
        except FileNotFoundError:
            print(f"Error: the file '{self.docmap_path}' does not exist.")
        try:
            with open(self.term_frequencies_path, "rb") as term_frequencies_file:
                term_frequencies = pickle.load(term_frequencies_file)
                self.term_frequencies = term_frequencies
        except FileNotFoundError:
            print(f"Error: the file '{self.term_frequencies_path}' does not exist.")
        return index, docmap, term_frequencies

    def get_tf(self, doc_id, term) -> int:
        token = clean_tokens(term)
        if len(token) != 1:
            raise Exception("Invalid token. Term must be 1 token")
        return self.term_frequencies[doc_id][token[0]]    

    def get_bm25_idf(self, term: str) -> float:
        token = clean_tokens(term)
        if len(token) != 1:
            raise Exception("Invalid token. Term must be 1 token")
        token = token[0]
        N = len(self.docmap) # total number of couments
        df = len(self.index[token]) # document frequency
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

def build_command() -> None:
    print(f"Initalizing index...")
    inverted_index = InvertedIndex()
    print(f"Starting build...")
    inverted_index.build()
    print(f"Saving index...")
    inverted_index.save()
    print(f"Save complete!")

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    index, docmap, tf = inverted_index.load()
    #movies = load_movies()
    results = []
    seen = set()
    query_tokens = clean_tokens(query)
    for token in query_tokens:
        if token not in index.keys():
            continue
        for doc in sorted(index[token]):
            if doc in seen:
                continue
            seen.add(doc)
            results.append(docmap[doc])
            if len(results) >= limit:
                return results

    return results

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    token = clean_tokens(term)
    token = token[0]
    tf = tf_command(doc_id, term)
    idf = idf_command(term)
    return tf * idf

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    token = clean_tokens(term)
    token = token[0]
    total_doc_count = len(idx.docmap)
    term_match_doc_count = len(idx.index[token])
    #print(f"total_doc_count: {total_doc_count}")
    #print(f"term_match_doc_count: {term_match_doc_count}")
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def bm25_idf_command(term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for q_token in query_tokens:
        for t_token in title_tokens:
            if q_token in t_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text


def tokenize(text: str) -> list:
    text = preprocess_text(text)
    tokens = text.split()
    tokens = list(filter(None, tokens))
    return tokens

def clean_tokens(text: str) -> list:
    stopwords = load_stopwords()
    tokens = tokenize(text)
    #tokens = stem_tokens(tokens)
    filtered_tokens = []
    for token in tokens:
        if token not in stopwords:
            filtered_tokens.append(token)
    tokens = stem_tokens(filtered_tokens)
    return tokens    


def stem_tokens(tokens: list) -> list:
    #used to reduce words to thier roots (ex running -> run)
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens





