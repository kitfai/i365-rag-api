# D:/infra365/codes/rag-git/batch_process/main.py

# --- Project Imports ---
# Make sure the project root is in the python path to find the core module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.service.qdrant_service import QdrantRAGService


def main():
    """Main function to run the RAG query system."""
    try:
        # 1. Initialize the RAG Service. This will handle all the
        #    DB creation/update logic automatically.
        rag_service = QdrantRAGService()
    except ValueError as e:
        print(f"ERROR during service initialization: {e}")
        return

    print("\n--- RAG System Ready. Ask your questions. Type 'exit' to quit. ---")

    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        print("Thinking...")
        # 2. Use the service's query method
        result = rag_service.query(question=query)

        print("\n--- Parent Documents Retrieved ---")
        if "context" in result and result["context"]:
            for i, doc in enumerate(result["context"]):
                source = doc.metadata.get('source', 'Unknown source')
                print(f"--- Document {i + 1}: (Source: {source}) ---")
                # For brevity, printing only the first 300 chars of the parent doc
                print(doc.page_content[:300] + "...")
                print("-" * 80)
        else:
            print("No documents were retrieved.")

        print("\nAnswer:")
        print(result["answer"])


if __name__ == "__main__":
    main()