# D:/infra365/codes/rag-git/core/service/qdrant_service.py

import pickle
from pathlib import Path
import torch
import logging
from operator import itemgetter # <-- Import itemgetter

# --- Core LangChain components ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import load_summarize_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# --- Vector Store and Embeddings ---
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import qdrant_client
from qdrant_client.http import models

# --- PDF Processing ---
from unstructured.partition.pdf import partition_pdf

# --- Configuration ---
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """Manages all configuration for the RAG service."""
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    LOG_FILE_PATH: Path = ROOT_DIR / "rag_service.log"
    PDFS_PATH: Path = ROOT_DIR / "batch_process" / "pdf"
    DB_PATH: Path = ROOT_DIR / "batch_process" / "vectorstore_qdrant"
    DOCSTORE_PATH: Path = DB_PATH / "docstore"

    QDRANT_COLLECTION_NAME: str = "rag_parent_documents"
    EMBEDDING_MODEL_NAME: str = 'BAAI/bge-large-en-v1.5'
    LLM_MODEL: str = 'deepseek-r1:latest'

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = RagSettings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=settings.LOG_FILE_PATH,
    filemode='a'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


class QdrantRAGService:
    """
    A service class to encapsulate all RAG logic including indexing,
    retrieval, and querying.
    """

    def __init__(self, force_rebuild: bool = False):
        """
        Initializes the service, setting up the retriever and the RAG chain.
        """
        logging.info("--- Initializing Qdrant RAG Service ---")
        logging.info(f"Using Qdrant DB path: {settings.DB_PATH}")

        self.retriever = self._get_or_create_retriever(force_rebuild)

        llm = Ollama(model=settings.LLM_MODEL)

        # Prompts remain the same
        initial_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert financial and analytical assistant.
            Based ONLY on the following document context, please answer the user's question.

            <context>
            {context}
            </context>

            Question: {input}
            """
        )
        refine_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert financial and analytical assistant.
            The user's original question is: {input}.
            You have already provided an existing answer: {existing_answer}.

            We have new context from another document below. If this new context adds
            any relevant information, refine the original answer. Otherwise, return the
            original answer. Do not make up information. Base your response ONLY on the
            provided context.

            <context>
            {context}
            </context>
            """
        )

        document_combination_chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=initial_prompt,
            refine_prompt=refine_prompt,
            document_variable_name="context",
        )

        # --- WORLD-CLASS FIX: Simplify the chain to a single, direct step ---
        self.rag_chain = (
            {
                # This dictionary creates the exact input needed by the final chain.
                # "input_documents" is populated by retrieving documents using the question.
                "input_documents": itemgetter("input") | self.retriever,
                # "input" is populated by passing the original question through.
                "input": itemgetter("input"),
            }
            # The output of the above is {"input_documents": [docs], "input": "user question"}
            # This is now in the perfect format to be piped directly into the final chain.
            | document_combination_chain
        )

        logging.info("--- RAG Service is ready. ---")

    def query(self, question: str) -> dict:
        """
        Performs a query against the RAG chain.
        """
        if not question:
            return {"answer": "Please provide a question.", "context": []}

        logging.info(f"Service querying with: '{question}'")
        result = self.rag_chain.invoke({"input": question})

        if result and result.get("output_text"):
            logging.info("Successfully generated answer.")
        else:
            logging.warning("Chain did not produce an answer.")

        return {
            "answer": result.get("output_text", "No answer could be generated."),
            "context": result.get("input_documents", [])
        }


    def _process_pdf_to_markdown(self, pdf_path: Path) -> str | None:
        """Processes a single PDF into a single Markdown string."""
        logging.info(f"Processing {pdf_path.name} into Markdown...")
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
            logging.error(f"Error processing {pdf_path.name}: {e}")
            return None

    def _get_or_create_retriever(self, force_rebuild: bool) -> ParentDocumentRetriever:
        """
        Builds/updates the vector store and initializes the ParentDocumentRetriever.
        """
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"--- Using device: {device} for embeddings ---")

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )

        qdrant_client_instance = qdrant_client.QdrantClient(path=str(settings.DB_PATH))

        collections_response = qdrant_client_instance.get_collections()
        collection_exists = any(c.name == settings.QDRANT_COLLECTION_NAME for c in collections_response.collections)

        pdfs_to_process = []
        if force_rebuild or not collection_exists:
            logging.info("--- Building new Vector DB and Docstore ---")
            settings.DB_PATH.mkdir(exist_ok=True)
            vector_size = len(embeddings.embed_query("test query"))
            qdrant_client_instance.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            pdfs_to_process = list(settings.PDFS_PATH.glob("*.pdf"))
            if not pdfs_to_process:
                raise ValueError("No PDF documents found to create a new database.")
        else:
            logging.info("--- Checking for new documents to update existing DB ---")
            try:
                points, _ = qdrant_client_instance.scroll(collection_name=settings.QDRANT_COLLECTION_NAME, limit=10000,
                                                          with_payload=True)
                indexed_files = {Path(p.payload['metadata']['source']).name for p in points if
                                 'metadata' in p.payload and 'source' in p.payload['metadata']}
            except Exception as e:
                logging.warning(f"Could not retrieve existing documents from Qdrant: {e}. Assuming DB is empty.")
                indexed_files = set()

            all_pdf_files_on_disk = {p.name for p in settings.PDFS_PATH.glob("*.pdf")}
            new_files_to_add = all_pdf_files_on_disk - indexed_files

            if not new_files_to_add:
                logging.info("No new documents to add. Vector DB is up to date.")
            else:
                logging.info(f"Found {len(new_files_to_add)} new documents to add.")
                pdfs_to_process = [settings.PDFS_PATH / name for name in new_files_to_add]

        vectorstore = QdrantVectorStore(
            client=qdrant_client_instance,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            # CRITICAL FIX: The parameter name must be plural.
            embedding=embeddings,
        )

        store = EncoderBackedStore(
            LocalFileStore(root_path=str(settings.DOCSTORE_PATH)),
            lambda key: key, pickle.dumps, pickle.loads
        )

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            id_key="doc_id"
        )

        if not pdfs_to_process:
            return retriever

        all_parent_docs = []
        for pdf_file in pdfs_to_process:
            content = self._process_pdf_to_markdown(pdf_file)
            if content:
                doc_id = str(pdf_file.name)
                doc = Document(page_content=content, metadata={"source": str(pdf_file), "doc_id": doc_id})
                all_parent_docs.append(doc)

        if all_parent_docs:
            logging.info(f"Adding {len(all_parent_docs)} documents to the retriever...")
            retriever.add_documents(all_parent_docs, ids=None)
            logging.info("--- DB and Docstore update complete. ---")

        return retriever