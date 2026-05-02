import string

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = clean_tokens(query)
        title_tokens = clean_tokens(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
        if len(results) >= limit:
            break
    return results


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
    tokens = text.split(" ")
    tokens = list(filter(None, tokens))
    return tokens

def clean_tokens(text: str) -> list:
    stopwords = load_stopwords()
    tokens = tokenize(text)
    tokens = stem_tokens(tokens)
    for token in tokens:
        if token in stopwords:
            tokens.remove(token)
    return tokens    


def stem_tokens(tokens: list) -> list:
    #used to reduce words to thier roots (ex running -> run)
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens





