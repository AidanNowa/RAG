import argparse
import json
import pickle

from lib.keyword_search import search_command, build_command, tf_command, idf_command, tfidf_command, bm25_idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    build_parser = subparsers.add_parser("build", help="Build inverted index")    

    search_parser = subparsers.add_parser("search", help="Search movies using BM52")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Return term frequency for a given doc_id and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search in document")

    idf_parser = subparsers.add_parser("idf", help="Return inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to view idf value for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Return TF-IDF for a given doc_id and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to view TF-IDF value for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")    
            results = search_command(args.query)
            #print(results)
            for i, result in enumerate(results, 1):
                #print(f"{i}. {result}")
                print(f"{i}. ({result['id']}) {result['title']}")
        case "build":
           # print(f"Initalizing index...")
           # inverted_index = InvertedIndex()
           # print(f"Starting build...")
           # inverted_index.build()
           # print(f"Saving index...")
           # inverted_index.save()
           # print(f"Save complete!")
            build_command()
            print("Inverted Index build successfully.")
            #with open('cache/index.pkl', 'rb') as index_file:
            #    docs = pickle.load(index_file)
            #print(f"DEBUG: First document for token 'merida' = {docs['merida']}")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
            #inverted_index = InvertedIndex()
            #index, docmap, tf = inverted_index.load()
            #print(inverted_index.get_tf(args.doc_id, args.term))
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
