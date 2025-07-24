# D:/infra365/codes/rag-git/core/service/qdrant_service.py

import pickle
from pathlib import Path
import torch
import logging
import time
import asyncio
from typing import Dict, List, Literal, Optional

# --- LangChain & Document Processing ---
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from unstructured.partition.auto import partition
from langchain.retrievers import ParentDocumentRetriever

# --- Qdrant & Configuration ---
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Define Document Types ---
DocType = Literal["Invoice", "Interest Advice", "Billing", "Unknown"]


class RagSettings(BaseSettings):
    """Manages all configuration for the RAG service."""
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    LOG_FILE_PATH: Path = ROOT_DIR / "rag_service.log"
    PDFS_PATH: Path = ROOT_DIR / "batch_process" / "pdf"
    DOCSTORE_PATH: Path = ROOT_DIR / "vectorstore_qdrant" / "docstore"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "rag_parent_documents"
    EMBEDDING_MODEL_NAME: str = 'BAAI/bge-large-en-v1.5'

    # Using a stable, instruction-following model is crucial.
    LLM_MODEL: str = 'llama3:latest'
    # It's recommended to use the same stable model for classification.
    LLM_CLASSIFIER_MODEL: str = 'llama3:latest'

    LLM_TIMEOUT: int = 360
    LLM_TEMPERATURE: float = 0.0
    LLM_CONTEXT_WINDOW: int = 4096
    LLM_MAX_NEW_TOKENS: int = 2048
    LLM_MIROSTAT: int = 2
    LLM_TOP_K: int = 40
    LLM_TOP_P: float = 0.9
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = RagSettings()

# --- Setup Logging ---
# Set level to DEBUG to see the full extracted text from PDFs in the log file.
# In production, you would set this to logging.INFO.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=settings.LOG_FILE_PATH,
    filemode='a'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Keep console output clean
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


class QdrantRAGService:
    """
    A high-performance RAG service using a ParentDocumentRetriever for full-context answers,
    LLM-based classification, and content-aware chunking. Implemented as a Singleton.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(QdrantRAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self, force_rebuild: bool = False):
        if self._initialized:
            return

        logging.info("--- Initializing Qdrant RAG Service (Singleton Instance) ---")

        self.qdrant_client = qdrant_client.QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            timeout=settings.LLM_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            num_ctx=settings.LLM_CONTEXT_WINDOW,
            num_predict=settings.LLM_MAX_NEW_TOKENS,
            # Add model-specific stop tokens for better generation control.
            stop=["<|endoftext|>", "##", "<|eot_id|>"],
            mirostat=settings.LLM_MIROSTAT,
            top_k=settings.LLM_TOP_K,
            top_p=settings.LLM_TOP_P
        )

        self.classifier_llm = OllamaLLM(
            model=settings.LLM_CLASSIFIER_MODEL,
            timeout=30,
            temperature=0.0
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        self._setup_qdrant_collection(force_rebuild)

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=self.embeddings,
        )

        self.docstore = EncoderBackedStore(
            LocalFileStore(str(settings.DOCSTORE_PATH)),
            lambda key: key, pickle.dumps, pickle.loads
        )

        # This splitter is used to create small, searchable chunks from the parent documents.
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # The ParentDocumentRetriever is the core of our advanced RAG strategy.
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
        )

        self._build_question_answer_chain()

        self._initialized = True
        logging.info("--- RAG Service is ready for queries. ---")

    def _build_question_answer_chain(self):
        """Builds the final question-answering part of the RAG chain."""
        prompt_template = """You are a precise data extraction engine for INFRA365 SDN BHD.
        Your task is to extract specific facts from the provided <context> to answer the <question>.

        **Crucial Rules:**
        - **Matching:** You must find the specific financial term requested (e.g., "Loan Amount", "Amount Billed"). Do not substitute terms. If the exact term is not found, the information is considered not available.
        - **Context-Only:** All information MUST come from the provided <context>. Do not use outside knowledge or make assumptions.
        - **Failure Condition:** If the requested information is not in the <context>, your ONLY response MUST be: "I cannot find the information in the provided documents."
        - **Output Format:** Your entire response MUST start with `## Summary Answer`. No text, reasoning, or thoughts are allowed before it.

        ---
        <context>
        {context}
        </context>

        ---
        <question>
        {input}
        </question>

        ---
        <answer>
        ## Summary Answer
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

    def _setup_qdrant_collection(self, force_rebuild: bool):
        collection_exists = False
        try:
            self.qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
            collection_exists = True
        except UnexpectedResponse as e:
            if e.status_code == 404:
                collection_exists = False
            else:
                raise e
        except Exception as e:
            raise e

        if force_rebuild or not collection_exists:
            logging.info(f"Recreating Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")
            vector_size = len(self.embeddings.embed_query("test"))
            self.qdrant_client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            self.qdrant_client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="metadata.doc_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _parse_llm_output(self, raw_answer: str) -> str:
        """
        Parses the raw output from the LLM, returning only the user-facing answer.
        """
        if "## Summary Answer" in raw_answer:
            # This handles the expected, correct output
            clean_part = raw_answer.split("## Summary Answer", 1)[1]
            return "## Summary Answer" + clean_part.strip()

        # This is a fallback for when the model fails to follow the format
        logging.warning("Could not find '## Summary Answer' delimiter in LLM output. Returning raw answer.")
        return raw_answer.strip()

    async def query(self, question: str, doc_type: Optional[str] = None) -> dict:
        """
        Performs a query using the ParentDocumentRetriever for full context.
        """
        if not question:
            return {"answer": "Please provide a question.", "context": []}

        logging.info(f"Service querying with: '{question}'")
        query_start_time = time.time()

        # Note: The standard ParentDocumentRetriever doesn't directly support metadata filtering
        # during the initial vector search. It retrieves based on vector similarity first, then
        # fetches the parent documents. For strict filtering, a custom retriever would be needed.
        if doc_type and doc_type != "Unknown":
            logging.warning(f"Note: doc_type filter '{doc_type}' is not directly applied with ParentDocumentRetriever.")

        retrieval_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

        logging.info("Invoking RAG chain with ParentDocumentRetriever...")
        invocation_start_time = time.time()
        result = await retrieval_chain.ainvoke({"input": question})
        invocation_end_time = time.time()
        logging.info(f" -> RAG chain invocation took {invocation_end_time - invocation_start_time:.2f} seconds.")

        raw_answer = result.get("answer", "No answer could be generated.")
        logging.info(f"Raw response from LLM:\n---\n{raw_answer}\n---")
        clean_answer = self._parse_llm_output(raw_answer)

        query_end_time = time.time()
        logging.info(f"Total query processing time: {query_end_time - query_start_time:.2f} seconds.")

        # Format the source documents for a cleaner final API response
        source_docs_formatted = [
            {"source": doc.metadata.get("source"), "page_content": doc.page_content}
            for doc in result.get("context", [])
        ]

        return {
            "answer": clean_answer,
            "source_documents": source_docs_formatted
        }

    async def process_new_documents(self):
        """Public method to run the async document processing task."""
        logging.info("Starting document ingestion process...")
        await self._process_and_upload_documents()
        logging.info("--- Document ingestion has finished. ---")

    async def _classify_document_type(self, content: str) -> DocType:
        valid_types = ", ".join([t for t in DocType.__args__ if t != "Unknown"])
        prompt = ChatPromptTemplate.from_template(
            "You are a document classification expert. Your task is to analyze the text provided and identify its type.\n"
            "Based on the following text, classify the document into ONE of the following types:\n"
            f"{valid_types}\n\n"
            "Respond with ONLY the type name from the list above. If you are unsure or the document does not match any type, respond with 'Unknown'.\n\n"
            "--- Document Text Sample ---\n"
            "{text_sample}\n"
            "--- End of Sample ---\n\n"
            "Classification:"
        )
        chain = prompt | self.classifier_llm | StrOutputParser()
        response_text = await chain.ainvoke({"text_sample": content[:2000]})

        # Clean up the response for more reliable matching
        cleaned_response = response_text.lower().strip()

        for doc_type_option in DocType.__args__:
            if doc_type_option.lower() == cleaned_response:
                return doc_type_option

        # Fallback for cases where the model might be slightly verbose
        for doc_type_option in DocType.__args__:
            if doc_type_option.lower() in cleaned_response:
                logging.warning(
                    f"Classification result '{response_text}' was not an exact match. Using fuzzy match: '{doc_type_option}'")
                return doc_type_option
        return "Unknown"

    def _process_pdf_to_markdown(self, pdf_path: Path) -> Optional[str]:
        logging.info(f" -> Starting markdown extraction for {pdf_path.name}")
        try:
            elements = partition(
                filename=str(pdf_path),
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng"],
                output_format="markdown"
            )
            markdown_content = "\n\n".join([el.text for el in elements])

            # This will print the full extracted text to your log file for verification.
            logging.debug(f"--- Extracted Markdown for {pdf_path.name} ---\n{markdown_content}\n--- End Markdown ---")

            logging.info(f" -> Successfully extracted markdown from {pdf_path.name}")
            return markdown_content if markdown_content.strip() else None
        except Exception as e:
            logging.error(f" -> Error processing {pdf_path.name} to markdown: {e}", exc_info=True)
            return None

    async def _process_single_pdf(self, pdf_file: Path, indexed_files: set):
        if pdf_file.name in indexed_files:
            return None
        logging.info(f"  -> Spawning worker thread for: {pdf_file.name}")
        markdown_content = await asyncio.to_thread(self._process_pdf_to_markdown, pdf_file)
        if not markdown_content:
            return None
        logging.info(f"  -> Markdown content received for {pdf_file.name}. Classifying document type...")
        try:
            doc_type = await self._classify_document_type(markdown_content)
            logging.info(f"  -> Classified {pdf_file.name} as type: {doc_type}. Now adding to retriever.")

            parent_doc = LangchainDocument(
                page_content=markdown_content,
                metadata={"source": str(pdf_file), "doc_type": doc_type}
            )

            # This single method handles splitting, embedding, and storing both parent and child docs.
            await self.retriever.aadd_documents([parent_doc], ids=None)

            logging.info(f"  -> DONE: Successfully processed and stored {pdf_file.name}.")
            return pdf_file.name
        except Exception as e:
            logging.error(f"  -> FAILED to process {pdf_file.name} after content extraction: {e}", exc_info=True)
            return None

    async def _process_and_upload_documents(self):
        logging.info("Step 1/3: Checking for already indexed files in Qdrant...")
        try:
            # A small optimization to only fetch the metadata field we need, not the vectors.
            points, _ = self.qdrant_client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                limit=10000, with_payload=["metadata.source"], with_vectors=False
            )
            # Safer way to build the set of indexed files
            indexed_files = {Path(p.payload['metadata']['source']).name for p in points if
                             p.payload and 'source' in p.payload.get('metadata', {})}
            logging.info(f" -> Found {len(indexed_files)} already indexed files.")
        except Exception:
            logging.warning(
                " -> Could not retrieve indexed files, assuming none. This is normal if the collection is new.")
            indexed_files = set()

        logging.info("Step 2/3: Discovering PDF files in the source directory...")
        all_pdf_files = list(settings.PDFS_PATH.glob("*.pdf"))
        logging.info(f" -> Found {len(all_pdf_files)} total PDF files.")
        pdfs_to_process = [pdf for pdf in all_pdf_files if pdf.name not in indexed_files]
        if not pdfs_to_process:
            logging.info(" -> All documents are already processed and up to date.")
            return

        logging.info(f" -> Found {len(pdfs_to_process)} new documents to process.")
        logging.info("Step 3/3: Processing new documents in parallel...")
        tasks = [self._process_single_pdf(pdf_file, indexed_files) for pdf_file in pdfs_to_process]
        results = await asyncio.gather(*tasks)
        processed_count = sum(1 for r in results if r is not None)
        logging.info(f" -> Parallel processing complete. Successfully added {processed_count} new documents.")