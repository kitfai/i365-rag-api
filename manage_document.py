# manage_document.py

import argparse
import json
from qdrant_client import QdrantClient, models

# --- Configuration ---
# These should match the settings of your Qdrant instance.
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# This is the name of the collection in Qdrant where your documents are stored.
# You will need to replace this with your actual collection name.
COLLECTION_NAME = "my_rag_collection"


def manage_document(client: QdrantClient, collection_name: str, document_name: str, reconstruct: bool, output_file: str):
    """
    Queries a Qdrant collection for a specific document.
    It can either display the metadata of all its chunks or reconstruct the full document text.

    Args:
        client: An initialized QdrantClient instance.
        collection_name: The name of the collection to search in.
        document_name: The name of the document to find.
        reconstruct: If True, reconstructs the document text from its chunks.
        output_file: If provided, saves the reconstructed text to this file.
    """
    print(f"🔍 Accessing document: '{document_name}' in collection '{collection_name}'...")

    try:
        # Scroll is the most efficient way to retrieve all points matching a filter.
        # We assume the document name is stored in a payload field called 'source'.
        # If you use a different field name, you must change the 'key' below.
        scroll_response, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",  # <-- IMPORTANT: Change this if your field name is different
                        match=models.MatchValue(value=document_name),
                    )
                ]
            ),
            limit=1000,  # Increase limit to fetch all chunks of a large document
            with_payload=True,
            with_vectors=False,
        )

        if not scroll_response:
            print(f"\n❌ No data found for document: '{document_name}'")
            return

        print(f"\n✅ Found {len(scroll_response)} chunks for '{document_name}'.")

        # If --reconstruct or --output-file is used, reconstruct the document.
        # Otherwise, just display the metadata.
        if reconstruct or output_file:
            reconstruct_document_text(scroll_response, output_file)
        else:
            display_metadata(scroll_response)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check the following:")
        print(f"1. Is the Qdrant service running and accessible at {QDRANT_HOST}:{QDRANT_PORT}?")
        print(f"2. Does the collection '{collection_name}' exist?")
        print("3. Is the payload field for the document name correct? (Currently set to 'source')")

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
    """
    Sorts document chunks and reconstructs the full text.

    Args:
        points: A list of Qdrant points (chunks) belonging to the document.
        output_file: The path to save the reconstructed text.
    """
    print("\nReconstructing parent document text...")

    # --- Sorting Logic ---
    # This is a critical step. We sort chunks to put them back in the correct order.
    # We'll try to sort by 'page_number' and then by 'chunk_index' if it exists.
    # If these keys are not in your payload, the text might be out of order.
    try:
        points.sort(key=lambda p: (
            p.payload.get('page_number', 0),
            p.payload.get('chunk_index', 0) # Assumes a chunk index within the page
        ))
        print("ℹ️ Sorted chunks by 'page_number' and 'chunk_index'.")
    except TypeError:
        print("⚠️ Warning: Could not sort chunks as 'page_number' or 'chunk_index' might be missing. Text may be out of order.")

    # We assume the main text content is in a payload field called 'text'.
    # Change 'text' if your field name is different.
    full_text = "\n\n".join([point.payload.get('text', '') for point in points])

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"✅ Successfully saved reconstructed document to '{output_file}'")
        except IOError as e:
            print(f"❌ Error saving file: {e}")
    else:
        # Print the reconstructed text to the console
        print("-" * 40)
        print(full_text)
        print("-" * 40)

def main():
    """Main function to parse command-line arguments and run the query."""
    parser = argparse.ArgumentParser(
        description="Query or reconstruct a parent document from a Qdrant RAG collection."
    )
    parser.add_argument(
        "document_name",
        type=str,
        help="The name of the document to query (e.g., 'my_document.pdf')."
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

    args = parser.parse_args()

    # Initialize the Qdrant client
    client = QdrantClient(host=args.host, port=args.port)

    manage_document(
        client=client,
        collection_name=args.collection,
        document_name=args.document_name,
        reconstruct=args.reconstruct,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()