import pickle
import os

from .keyword_search import clean_tokens, load_movies
from .search_utils import CACHE_DIR

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
    
    def __add_document(self, doc_id, text) -> None:
        tokens = clean_tokens(text)
        for token in tokens:
            if token in self.index.keys():
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id]) #create a set of doc_ids to add to
        
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




       
