import logging
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path to find the core module
# This ensures that 'from core.service...' works correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.service.qdrant_service import QdrantRAGService

# Make the main function asynchronous so we can use 'await'
async def main():
    """
    This script is dedicated to DOCUMENT INGESTION.
    It initializes the RAG service and runs the document processing pipeline.
    """
    # --- Control Database Rebuilds Here ---
    # Set to True only when you want to clear and re-process all documents.
    # Set to False to only process new, unprocessed documents.
    REBUILD_DB = True
    # ------------------------------------

    if REBUILD_DB:
        logging.info("--- Force Rebuild is ON. The database collection will be wiped and recreated. ---")
    else:
        logging.info("--- Force Rebuild is OFF. Only new documents will be processed. ---")

    # 1. Initialize the service.
    # The force_rebuild flag will ensure the collection is either cleared or ready.
    #rag_service = QdrantRAGService(force_rebuild=REBUILD_DB)
    rag_service = QdrantRAGService()

    # 2. Explicitly run the document processing and ingestion pipeline.
    # This will find all PDFs, process them, and upload them to the fresh collection.
    #await rag_service.process_new_documents()
    #rag_service = QdrantRAGService()
    # --- Define Your Test Query Here ---
    test_question = "List out all the details of billings billed to MR MUHAMEED AZAM BIN ALISAN"
    test_doc_type = "Invoice"
    # ---------------------------------

    logging.info(f"--- Running test query: '{test_question}' ---")

    # Await the asynchronous query method. This is the crucial fix.
    result = await rag_service.query(question=test_question, doc_type=test_doc_type)

    # Now 'result' will be the dictionary we expect.
    print("\n--- Query Result ---")
    print(f"\nAnswer:\n{result.get('answer')}")

    if "context" in result and result["context"]:
        print("\n--- Source Documents Retrieved ---")
        for i, doc in enumerate(result["context"]):
            print(f"  {i+1}. Source: {doc.metadata.get('source')}")
            # Optionally print a snippet of the content
            # print(f"     Content: {doc.page_content[:150].replace('\n', ' ')}...")
    else:
        print("\n--- No Source Documents Found ---")

# Use asyncio.run() to execute the async main function from your script
if __name__ == "__main__":
    asyncio.run(main())