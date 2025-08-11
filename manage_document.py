# manage_document.py

import argparse
import json
from qdrant_client import QdrantClient, models

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "my_rag_collection"


def manage_document(client: QdrantClient, collection_name: str, reconstruct: bool, output_file: str, document_name: str = None, document_id: str = None):
    """
    Queries a Qdrant collection for a specific document by its name or ID.
    It can either display the metadata of all its chunks or reconstruct the full document text.
    """
    # --- CODE IMPROVEMENT: Dynamic Query Logic ---
    # Determine the query key and value based on the provided identifier.
    if document_id:
        query_key = "metadata.doc_id"
        query_value = document_id
        identifier_type = "ID"
        print(f"🔍 Accessing document by ID: '{document_id}' in collection '{collection_name}'...")
    elif document_name:
        query_key = "metadata.filename"
        query_value = document_name
        identifier_type = "filename"
        print(f"🔍 Accessing document by filename: '{document_name}' in collection '{collection_name}'...")
    else:
        # This case should be prevented by the main function's logic.
        print("Error: No document identifier (name or ID) was provided.")
        return

    try:
        scroll_response, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=query_key,
                        match=models.MatchValue(value=query_value),
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        if not scroll_response:
            # --- ENHANCED ERROR MESSAGE ---
            print(f"\n❌ No data found for document where {query_key} = '{query_value}'")
            print("\n--- DEBUGGING SUGGESTIONS ---")
            print(f"1. Data Mismatch: Ensure the document was ingested with a '{identifier_type}' field in its metadata.")
            print(f"2. Typo Check: Double-check the provided {identifier_type} and the collection name ('{collection_name}').")
            print("\n💡 Tip: Use the --peek-metadata flag to inspect the actual data structure in your collection.")
            return

        print(f"\n✅ Found {len(scroll_response)} chunks for the document.")

        if reconstruct or output_file:
            reconstruct_document_text(scroll_response, output_file)
        else:
            display_metadata(scroll_response)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check the following:")
        print(f"1. Is the Qdrant service running and accessible at {QDRANT_HOST}:{QDRANT_PORT}?")
        print(f"2. Does the collection '{collection_name}' exist?")
        print(f"3. Is the payload field for the query correct? (Currently set to '{query_key}')")



def display_metadata(points: list):
    """Prints the metadata for a list of Qdrant points."""
    print("Displaying chunk metadata:")
    print("-" * 40)
    for i, point in enumerate(points):
        print(f"--- Chunk {i + 1} (Point ID: {point.id}) ---")
        metadata = point.payload
        print(json.dumps(metadata, indent=4))
        print()


def reconstruct_document_text(points: list, output_file: str):
    """Sorts document chunks and reconstructs the full text."""
    print("\nReconstructing parent document text...")
    try:
        points.sort(key=lambda p: (
            p.payload.get('metadata', {}).get('page_number', 0),
            p.payload.get('metadata', {}).get('chunk_index', 0)
        ))
        print("ℹ️ Sorted chunks by 'metadata.page_number' and 'metadata.chunk_index'.")
    except TypeError:
        print("⚠️ Warning: Could not sort chunks. Text may be out of order.")

    full_text = "\n\n".join([point.payload.get('page_content', '') for point in points])

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"✅ Successfully saved reconstructed document to '{output_file}'")
        except IOError as e:
            print(f"❌ Error saving file: {e}")
    else:
        print("-" * 40)
        print(full_text)
        print("-" * 40)


def peek_at_metadata(client: QdrantClient, collection_name: str, limit: int = 5):
    """Fetches and displays the payload of a few random points from the collection for debugging."""
    print(f"🕵️ Peeking at the metadata of {limit} documents in '{collection_name}'...")
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        if not points:
            print("Collection is empty.")
            return

        print("-" * 40)
        for i, point in enumerate(points):
            print(f"--- Sample Document {i + 1} (Point ID: {point.id}) ---")
            print(json.dumps(point.payload, indent=4))
            print()
        print("-" * 40)
        print("Review the payload structure above. Does it contain a 'metadata' object with a 'filename' field?")

    except Exception as e:
        print(f"\nAn error occurred while peeking: {e}")


def main():
    """Main function to parse command-line arguments and run the query."""
    parser = argparse.ArgumentParser(
        description="Query or reconstruct a parent document from a Qdrant RAG collection."
    )
    parser.add_argument(
        "document_name",
        nargs='?',
        default=None,
        type=str,
        help="The name of the document to query (e.g., 'my_document.pdf'). Required unless --peek-metadata is used."
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Reconstruct the full text of the parent document and print it to the console."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        metavar="FILE_PATH",
        help="Save the reconstructed document text to the specified file."
    )
    parser.add_argument(
        "--peek-metadata",
        action="store_true",
        help="Show the metadata of a few random documents to debug the payload structure."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"The name of the Qdrant collection (default: {COLLECTION_NAME})."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=QDRANT_HOST,
        help=f"The hostname or IP of the Qdrant instance (default: {QDRANT_HOST})."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=QDRANT_PORT,
        help=f"The port of the Qdrant instance (default: {QDRANT_PORT})."
    )

    # --- CODE IMPROVEMENT: Added --by-id flag ---
    parser.add_argument(
        "--by-id",
        type=str,
        metavar="DOCUMENT_ID",
        help="Query by the unique document ID (UUID) instead of the filename."
    )

    args = parser.parse_args()
    client = QdrantClient(host=args.host, port=args.port)

    if args.peek_metadata:
        peek_at_metadata(client, args.collection)
    elif args.document_name or args.by_id:
        if args.document_name and args.by_id:
            parser.error("Please provide either a document_name or use --by-id, but not both.")

        manage_document(
            client=client,
            collection_name=args.collection,
            document_name=args.document_name,
            document_id=args.by_id,
            reconstruct=args.reconstruct,
            output_file=args.output_file
        )
    else:
        parser.error("You must provide a document_name, or use --by-id, or use the --peek-metadata flag.")


if __name__ == "__main__":
    main()