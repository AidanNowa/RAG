import argparse
import json
import pickle

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    build_parser = subparsers.add_parser("build", help="Build inverted index")    

    search_parser = subparsers.add_parser("search", help="Search movies using BM52")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")    
            results = search_command(args.query)
            for i, result in enumerate(results, 1):
                #print(f"{i}. {result}")
                print(f"{i}. ({result['id']}) {result['title']}")
        case "build":
            print(f"Initalizing index...")
            inverted_index = InvertedIndex()
            print(f"Starting build...")
            inverted_index.build()
            print(f"Saving index...")
            inverted_index.save()
            print(f"Save complete!")
            with open('cache/index.pkl', 'rb') as index_file:
                docs = pickle.load(index_file)
            print(f"DEBUG: First document for token 'merida' = {docs['merida']}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
