# D:/infra365/codes/rag-git/core/service/qdrant_service.py

import pickle
from pathlib import Path
import torch
import logging
import time
import asyncio
from typing import Dict, List, Literal, Optional
import re

# --- LangChain & Document Processing ---
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from unstructured.partition.auto import partition
from langchain.retrievers import ParentDocumentRetriever
from langchain_openai import ChatOpenAI

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
    RETRIEVER_TOP_K: int = 3  # New setting, reduced from 5 to 3

    # --- VLLM Server Configuration ---
    # The model name as served by your vLLM instance
    LLM_MODEL: str = "llama-3-8b-instruct"
    # The base URL of your OpenAI-compatible vLLM server
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    # An API key is required, but can be a dummy value for local servers
    VLLM_API_KEY: str = "not-needed"
    # --- Classifier Model ---
    # For simplicity, we use the same vLLM endpoint for classification
    LLM_CLASSIFIER_MODEL: str = "llama-3-8b-instruct"

    # --- LLM Generation Parameters ---
    LLM_TIMEOUT: int = 360
    LLM_TEMPERATURE: float = 0.0
    LLM_CONTEXT_WINDOW: int = 4096
    LLM_MAX_NEW_TOKENS: int = 1024
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

        # Instantiate the ChatOpenAI client to connect to the vLLM server
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.VLLM_API_KEY,
            openai_api_base=settings.VLLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_NEW_TOKENS,
        )

        # Use the same vLLM endpoint for classification for consistency
        self.classifier_llm = ChatOpenAI(
            model=settings.LLM_CLASSIFIER_MODEL,
            openai_api_key=settings.VLLM_API_KEY,
            openai_api_base=settings.VLLM_BASE_URL,
            temperature=0.0,
            max_tokens=50,  # Classification needs very few tokens
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

        # Use the ParentDocumentRetriever directly for robustness and simplicity.
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
        )
        self.retriever.search_kwargs = {"k": settings.RETRIEVER_TOP_K}  # Fetch top 5 candidate documents

        self._build_question_answer_chain()

        self._initialized = True
        logging.info("--- RAG Service is ready for queries. ---")

    def _build_question_answer_chain(self):
        """Builds the final question-answering part of the RAG chain."""
        prompt_template = """You are a deterministic data extraction script. Your only function is to execute a checklist to find facts in the <context> that exactly match the <question>.
        **Your Process:**
        1.  **Deconstruct Question:** Break down the user's <question> into a checklist of required entities.
        2.  **Scan for Evidence:** Find a single, continuous block of text in the <context> that contains ALL the entities from your checklist.
        3.  **Verify Checklist:** In the <scratchpad>, you MUST fill out the verification checklist. For each item, you must state Yes or No.
        4.  **Synthesize Answer:** If AND ONLY IF all checklist items are "Yes", construct the answer. Otherwise, you must state that the information was not found.

        **Crucial Rules:**
        -   **Your only possible outputs are a structured answer following the format below, or the exact sentence 'I cannot find the information in the provided documents'. You are forbidden from outputting any other text, refusal, or explanation.**
        -   **NEVER** invent or assume information not explicitly present in the <context>.
        -   If the context does not contain the information for the specific entities requested, you MUST respond with ONLY the following sentence: "I cannot find the information in the provided documents."
        -   **Strict Association Rule:** If you find a "Loan Amount" for the correct person but the wrong project, the checklist fails and the information is NOT FOUND. All entities must match in the same document section.
        -   Your entire response MUST strictly follow the format defined in the **"Output Format"** section.

        ---
        <context>
        {context}
        </context>
        ---

        **Question:**
        {input}

        ---
        **Output Format:**

          <scratchpad>
          *You MUST fill out this checklist.
          1.  **Deconstructed Question:**
              -   Person/Entity: [Name from question]
              -   Project: [Project from question]
              -   Metric: [Data point from question]
          2.  **Evidence Snippet:** [Quote the single, most relevant text block from the context here]
          3.  **Verification Checklist:**
              -   Snippet contains Person/Entity? [Yes/No]
              -   Snippet contains Project? [Yes/No]
              -   Snippet contains Metric? [Yes/No]
          4.  **Conclusion:** [State 'All checks passed' or 'Checklist failed, information not found.']
          This section will not be shown in the final output.*
          </scratchpad>

        ## Summary Answer
        *Provide a concise, direct answer to the user's question based on your findings.*

        ## Detailed Breakdown
        *List every single piece of evidence you used to construct the summary answer. For each monetary amount, specify which document it came from.*
        -   Fact 1 from Source X
        -   Fact 2 from Source Y

        ## Source Documents
        *List the unique source documents you used to find the answer. List out the original file name used as reference*
        -   `source_document_1.pdf`
        -   `source_document_2.pdf`
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
        Performs a query. If a doc_type is specified, it uses a strict, filter-aware
        retriever. Otherwise, it uses the default advanced retriever.
        """
        if not question:
            return {"answer": "Please provide a question.", "source_documents": []}

        logging.info(f"Service querying with: '{question}'")
        query_start_time = time.time()

        # Store original search_kwargs to restore them later, ensuring thread safety.
        original_search_kwargs = self.retriever.search_kwargs.copy()
        active_retriever = self.retriever

        result = {}
        retrieved_docs = []

        try:
            # Dynamically inject the filter into the main retriever
            if doc_type and doc_type != "Unknown":
                logging.info(f"Applying strict doc_type filter: '{doc_type}'.")
                qdrant_filter = models.Filter(
                    must=[models.FieldCondition(key="metadata.doc_type", match=models.MatchValue(value=doc_type))]
                )
                active_retriever.search_kwargs["filter"] = qdrant_filter
            else:
                logging.info("No doc_type filter applied. Using default retriever.")

            # --- World-Class RAG: Separate retrieval from generation for better control ---
            # Step 1: Retrieve documents from the vector store.
            logging.info("Step 1: Retrieving relevant documents...")
            retrieved_docs = await active_retriever.ainvoke(question)

            # Step 2: Explicitly handle the case where no documents are found.
            if not retrieved_docs:
                logging.warning("No relevant documents found by the retriever for the given query and filter.")
                return {
                    "answer": "I cannot find the information in the provided documents.",
                    "source_documents": []
                }

            # Step 3: If documents are found, invoke the LLM with the context.
            logging.info(f"Step 2: Found {len(retrieved_docs)} documents. Invoking LLM chain...")
            invocation_start_time = time.time()

            raw_answer = await self.question_answer_chain.ainvoke({
                "input": question,
                "context": retrieved_docs
            })

            invocation_end_time = time.time()
            logging.info(f" -> RAG chain invocation took {invocation_end_time - invocation_start_time:.2f} seconds.")

            result = {"answer": raw_answer, "context": retrieved_docs}

        finally:
            # IMPORTANT: Restore the original search_kwargs to ensure this query
            # doesn't affect the next one.
            self.retriever.search_kwargs = original_search_kwargs

        raw_answer = result.get("answer", "No answer could be generated.")
        logging.info(f"Raw response from LLM:\n---\n{raw_answer}\n---")
        clean_answer = self._parse_llm_output(raw_answer)

        query_end_time = time.time()
        logging.info(f"Total query processing time: {query_end_time - query_start_time:.2f} seconds.")

        # Format the source documents for a cleaner final API response
        source_docs_formatted = [
            {"source": Path(doc.metadata.get("source")).name, "page_content": doc.page_content}
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
            "You are an automated document classification system. Your ONLY function is to output a single word from the provided list.\n"
            "Analyze the text and classify it into ONE of these types:\n"
            f"{valid_types}\n\n"
            "Your entire response MUST be a single word from that list. DO NOT add any explanation, preamble, or notes.\n"
            "If you are unsure, respond with the single word 'Unknown'.\n\n"
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
            # Use regex to find the type as a whole word, case-insensitive, for more accuracy
            if re.search(r'\b' + re.escape(doc_type_option.lower()) + r'\b', cleaned_response):
                logging.warning(
                    f"Classification result '{response_text}' was not an exact match. Using fuzzy match: '{doc_type_option}'")
                return doc_type_option

        logging.warning(
            f"Classifier could not determine a valid type for response: '{response_text}'. Defaulting to 'Unknown'.")
        return "Invoice"

    def _process_pdf_to_markdown(self, pdf_path: Path) -> Optional[str]:
        logging.info(f" -> Starting markdown extraction for {pdf_path.name}")
        try:
            # This resolves the deprecation warning from the unstructured library.
            elements = partition(
                filename=str(pdf_path),
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng"],
                output_format="markdown",
                size={"longest_edge": 2048}
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

            # The `add_documents` method of the ParentDocumentRetriever is synchronous.
            # We must run it in a separate thread to avoid blocking the asyncio event loop.
            await asyncio.to_thread(
                self.retriever.add_documents,
                [parent_doc],
                ids=None
            )

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