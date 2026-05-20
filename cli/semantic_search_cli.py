import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_text, semantic_chunk_text, embed_chunks_command

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify embedding model is loaded and its max sequence length")
    
    single_embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single given text")
    single_embed_text_parser.add_argument("text", type=str, help="Text to be embedded")    
   
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings for full document list")
 
    embed_query_parser = subparsers.add_parser("embed_query", help="Generate the embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Utilize semantic search to query a database based on input")
    search_parser.add_argument("query", type=str, help="Query to be searched")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Set number of results to return, default is 5")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk inputed text into fixed-size chunks with optional overlap")
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Set number of words in each chunk")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Set number of words to overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk inputted text into set sized chunks with optional overlap")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be semantically chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Set number of sentances in each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Set number of sentences to overlap between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate chunked semantic embeddings")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap) 
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
