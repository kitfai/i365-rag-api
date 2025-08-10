from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

# --- Configuration ---
# Replace with the name of your collection
COLLECTION_NAME = "rag_parent_documents"
# Your Qdrant instance is running on localhost based on your README
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def list_all_documents(client: QdrantClient, collection_name: str):
    """
    Iterates through all documents in a Qdrant collection and prints their
    ID and metadata (payload).

    Args:
        client: An initialized QdrantClient instance.
        collection_name: The name of the collection to scroll through.
    """
    print(f"--- Fetching all documents from collection: '{collection_name}' ---")

    try:
        # The 'scroll' method is the recommended way to iterate over all points.
        # - with_payload=True:  Crucial for fetching the metadata.
        # - with_vectors=False: Best practice to exclude the large vector data,
        #                       making the response much faster and smaller.
        # - limit=100:          Sets the page size for each scroll request.
        response, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        total_docs = 0
        while True:
            for point in response:
                print(f"Document ID: {point.id}")
                print(f"  Metadata: {point.payload}")
                print("-" * 20)
                total_docs += 1

            # If next_page_offset is None, we've reached the end
            if next_page_offset is None:
                break

            # Fetch the next page of results
            response, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=next_page_offset,
            )

        print(f"\n--- Found a total of {total_docs} documents. ---")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Please ensure the collection '{collection_name}' exists and Qdrant is running.")


if __name__ == "__main__":
    # Initialize the client to connect to your Qdrant instance
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    list_all_documents(qdrant_client, COLLECTION_NAME)
