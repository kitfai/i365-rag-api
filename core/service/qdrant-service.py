# D:/Codes/pdf-rag/rag-markdown-qrant.py

import pickle
from pathlib import Path

# --- Core LangChain components for Parent Document Retrieval ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Vector Store and Embeddings ---
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import qdrant_client
from qdrant_client.http import models

# --- RAG Chain and LLM components ---
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# --- PDF Processing ---
from unstructured.partition.pdf import partition_pdf

# --- Configuration ---
ROOT_DIR = Path(__file__).parent
PDFS_PATH = ROOT_DIR / "pdf"
DB_PATH = ROOT_DIR / "vectorstore_qdrant"
DOCSTORE_PATH = DB_PATH / "docstore"
QDRANT_COLLECTION_NAME = "rag_parent_documents"

# Embedding and LLM model configuration
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
LLM_MODEL = 'deepseek-r1:latest'


def process_pdf_to_markdown(pdf_path: Path) -> str | None:
    """Processes a single PDF into a single Markdown string."""
    print(f"Processing {pdf_path.name} into Markdown...")
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
            languages=["eng"],
            output_format="markdown"
        )
        markdown_content = "\n\n".join([el.text for el in elements])
        return markdown_content if markdown_content.strip() else None
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return None

'''
def get_retriever(force_rebuild: bool = False) -> ParentDocumentRetriever:
    """
    Initializes and returns a ParentDocumentRetriever with a persistent docstore.
    Handles building the DB for the first time and updating it with new documents.
    """
    # --- Initialize Core Components ---
    #child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    qdrant_client_instance = qdrant_client.QdrantClient(path=str(DB_PATH))

    # --- FIX 1: Correct the 'embeddings' keyword argument ---
    vectorstore = QdrantVectorStore(
        client=qdrant_client_instance,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings, # Was 'embedding='
    )

    # --- FIX 2: Provide all required positional arguments to EncoderBackedStore ---
    # 1. Create the base store that stores raw bytes
    base_store = LocalFileStore(root_path=str(DOCSTORE_PATH))
    # 2. Create a store that correctly serializes/deserializes objects
    store = EncoderBackedStore(
        base_store,
        lambda key: key,      # key_encoder: pass-through for string keys
        pickle.dumps,         # value_serializer: object -> bytes
        pickle.loads          # value_deserializer: bytes -> object
    )
    # --- End of FIX ---

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        id_key="doc_id"
    )

    # --- Smarter logic to check for and add new documents ---
    collections_response = qdrant_client_instance.get_collections()
    collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections_response.collections)

    pdfs_to_process = []
    if force_rebuild or not collection_exists:
        print("--- Building new Vector DB and Docstore ---")
        DB_PATH.mkdir(exist_ok=True)

        try:
            vector_size = len(embeddings.embed_query("test query"))
            print(f"Determined vector size: {vector_size}")
        except Exception as e:
            print(f"Could not determine vector size from embeddings, defaulting to 1024. Error: {e}")
            vector_size = 1024

        qdrant_client_instance.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{QDRANT_COLLECTION_NAME}' created/recreated in Qdrant.")

        pdfs_to_process = list(PDFS_PATH.glob("*.pdf"))
        if not pdfs_to_process:
            raise ValueError("No PDF documents found to create a new database.")
    else:
        print("--- Checking for new documents to update existing DB ---")
        try:
            points, _ = qdrant_client_instance.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10000,
                                                      with_payload=True)
            indexed_files = {Path(p.payload['metadata']['source']).name for p in points if
                             'metadata' in p.payload and 'source' in p.payload['metadata']}
        except Exception as e:
            print(f"Warning: Could not retrieve existing documents from Qdrant: {e}. Assuming DB is empty.")
            indexed_files = set()

        all_pdf_files_on_disk = {p.name for p in PDFS_PATH.glob("*.pdf")}
        new_files_to_add = all_pdf_files_on_disk - indexed_files

        if not new_files_to_add:
            print("No new documents to add. Vector DB is up to date.")
            return retriever

        print(f"Found {len(new_files_to_add)} new documents to add.")
        pdfs_to_process = [PDFS_PATH / name for name in new_files_to_add]

    # This block runs for both initial build and updates
    all_parent_docs = []
    for pdf_file in pdfs_to_process:
        content = process_pdf_to_markdown(pdf_file)
        if content:
            doc_id = str(pdf_file.name)
            doc = Document(page_content=content, metadata={"source": str(pdf_file), "doc_id": doc_id})
            all_parent_docs.append(doc)

    if all_parent_docs:
        print(f"Adding {len(all_parent_docs)} documents to the retriever...")
        retriever.add_documents(all_parent_docs, ids=None)
        print("--- DB and Docstore update complete. ---")

    return retriever
'''

def get_retriever(force_rebuild: bool = False) -> ParentDocumentRetriever:
    """
    Initializes and returns a ParentDocumentRetriever with a persistent docstore.
    Handles building the DB for the first time and updating it with new documents.
    """
    # --- Initialize Core Components ---
    #child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    qdrant_client_instance = qdrant_client.QdrantClient(path=str(DB_PATH))

    # --- Smarter logic to check for and add new documents ---
    collections_response = qdrant_client_instance.get_collections()
    collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections_response.collections)

    pdfs_to_process = []
    if force_rebuild or not collection_exists:
        print("--- Building new Vector DB and Docstore ---")
        DB_PATH.mkdir(exist_ok=True)

        try:
            vector_size = len(embeddings.embed_query("test query"))
            print(f"Determined vector size: {vector_size}")
        except Exception as e:
            print(f"Could not determine vector size from embeddings, defaulting to 1024. Error: {e}")
            vector_size = 1024

        qdrant_client_instance.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{QDRANT_COLLECTION_NAME}' created/recreated in Qdrant.")

        pdfs_to_process = list(PDFS_PATH.glob("*.pdf"))
        if not pdfs_to_process:
            raise ValueError("No PDF documents found to create a new database.")
    else:
        print("--- Checking for new documents to update existing DB ---")
        try:
            points, _ = qdrant_client_instance.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10000,
                                                      with_payload=True)
            indexed_files = {Path(p.payload['metadata']['source']).name for p in points if
                             'metadata' in p.payload and 'source' in p.payload['metadata']}
        except Exception as e:
            print(f"Warning: Could not retrieve existing documents from Qdrant: {e}. Assuming DB is empty.")
            indexed_files = set()

        all_pdf_files_on_disk = {p.name for p in PDFS_PATH.glob("*.pdf")}
        new_files_to_add = all_pdf_files_on_disk - indexed_files

        if not new_files_to_add:
            print("No new documents to add. Vector DB is up to date.")
        else:
            print(f"Found {len(new_files_to_add)} new documents to add.")
            pdfs_to_process = [PDFS_PATH / name for name in new_files_to_add]

    # --- FIX 1: Instantiate the vector store AFTER the collection is guaranteed to exist ---
    vectorstore = QdrantVectorStore(
        client=qdrant_client_instance,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings, # --- FIX 2: Corrected keyword from 'embedding' to 'embeddings'
    )

    # Create the docstore for parent documents
    base_store = LocalFileStore(root_path=str(DOCSTORE_PATH))
    store = EncoderBackedStore(
        base_store,
        lambda key: key,      # key_encoder: pass-through for string keys
        pickle.dumps,         # value_serializer: object -> bytes
        pickle.loads          # value_deserializer: bytes -> object
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        id_key="doc_id"
    )

    # If there are no new files to process (in the update case), we can return early.
    if not pdfs_to_process and not (force_rebuild or not collection_exists):
        return retriever

    # This block runs for both initial build and updates with new files
    all_parent_docs = []
    for pdf_file in pdfs_to_process:
        content = process_pdf_to_markdown(pdf_file)
        if content:
            doc_id = str(pdf_file.name)
            doc = Document(page_content=content, metadata={"source": str(pdf_file), "doc_id": doc_id})
            all_parent_docs.append(doc)

    if all_parent_docs:
        print(f"Adding {len(all_parent_docs)} documents to the retriever...")
        retriever.add_documents(all_parent_docs, ids=None)
        print("--- DB and Docstore update complete. ---")

    return retriever

def main():
    """Main function to run the RAG query system."""
    try:
        retriever = get_retriever()
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    prompt_template = """
    You are an expert financial and analytical assistant for 'Infra Mewah Development Sdn Bhd', a construction company.
    Your primary task is to provide precise answers by analyzing the provided documents, which are formatted in Markdown.

    **Your Process:**
    1.  Carefully read the user's question to understand the specific information required.
    2.  The <context> below contains the full text of one or more documents in **Markdown format**. Pay close attention to the structure.
    3.  Synthesize the information from the context to formulate your answer.
    4.  Formulate your final answer based **ONLY** on the information found in the <context>.

    **Crucial Rules:**
    -   If the answer is not in the context, you MUST state: "I cannot find the information in the provided documents."
    -   Do not make assumptions or use external knowledge.
    -   When providing a total amount, list the individual amounts and their sources (e.g., "RM 5,000 from receipt #R-001") before giving the final total.

    <context>
    {context}
    </context>

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = Ollama(model=LLM_MODEL)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n--- RAG System Ready. Ask your questions. Type 'exit' to quit. ---")

    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        print("Thinking...")
        result = rag_chain.invoke({"input": query})

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