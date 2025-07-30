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
from langchain_openai import ChatOpenAI
# --- Qdrant & Configuration ---
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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
    #LLM_MODEL: str = 'llama3:latest'
    #LLM_MODEL: str = 'deepseek-r1:latest'
    # It's recommended to use the same stable model for classification.
    #LLM_CLASSIFIER_MODEL: str = 'deepseek-r1:latest'
    # --- VLLM Server Configuration ---
    # The model name as served by your vLLM instance
    LLM_MODEL: str = "TheBloke/deepseek-coder-1.3B-instruct-AWQ"
    # The base URL of your OpenAI-compatible vLLM server
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    # An API key is required, but can be a dummy value for local servers
    VLLM_API_KEY: str = "not-needed"
    # --- Classifier Model ---
    # For simplicity, we use the same vLLM endpoint for classification
    LLM_CLASSIFIER_MODEL: str = "TheBloke/deepseek-coder-1.3B-instruct-AWQ"

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
        '''
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
        )'''
        # NEW: Instantiate the ChatOpenAI client to connect to the vLLM server
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
        '''
        self.classifier_llm = OllamaLLM(
            model=settings.LLM_CLASSIFIER_MODEL,
            timeout=30,
            temperature=0.0
        )
        '''
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

        base_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
        )
        base_retriever.search_kwargs = {"k": 5}  # Fetch 5 candidates


        # Create a compressor that uses a fast LLM to extract relevant sentences
        # This is a cheap but effective way to filter noise.
        compressor = LLMChainExtractor.from_llm(self.llm)

        #Create the final compression retriever
        # This will first call the base_retriever, then pass the results to the compressor.
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        #self.retriever.search_kwargs = {"k": 5}
        self._build_question_answer_chain()

        self._initialized = True
        logging.info("--- RAG Service is ready for queries. ---")

    def _build_question_answer_chain(self):
        """Builds the final question-answering part of the RAG chain."""
        prompt_template = """You are a specialized data extraction engine for 'INFRA365 SDN BHD'.
        Your sole purpose is to extract specific facts from the provided <context> to answer the user's <question>.

        **Your Process:**
        1.  **Analyze the Question:** First, understand what specific pieces of information the user is asking for.
        2.  **Scan the Context:** Systematically scan the entire <context> for the key entities (e.g., names like "LIEW CHIN GUAN", document types like "Interest Advice") and data points (e.g., monetary amounts, dates) mentioned in the question.
        3.  **Extract Verbatim:** Extract the relevant facts exactly as they appear in the documents. Do not interpret or summarize them at this stage.
        4.  **Synthesize the Answer:** Combine the extracted facts into a coherent answer, following the format below.

        **Crucial Rules:**
        -   **NEVER** invent or assume information not explicitly present in the <context>.
        -   If the context does not contain the necessary information to answer the question, you MUST respond with ONLY the following sentence: "I cannot find the information in the provided documents."
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
        *Use this space to think step-by-step. Outline your plan for finding the answer. Identify the key entities and data points you need to find. This section will not be shown in the final output.*
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
            return {"answer": "Please provide a question.", "context": []}

        logging.info(f"Service querying with: '{question}'")
        query_start_time = time.time()

        # Store original search_kwargs to restore them later, ensuring thread safety.
        original_search_kwargs = self.retriever.base_retriever.search_kwargs.copy()
        active_retriever = self.retriever

        try:
            # --- FIX: Dynamically inject the filter into the main retriever ---
            # This ensures we always use the powerful ParentDocumentRetriever,
            # providing full, high-quality context to the LLM.
            if doc_type and doc_type != "Unknown":
                logging.info(f"Applying strict doc_type filter: '{doc_type}'.")

                # Create the filter for Qdrant
                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.doc_type",
                            match=models.MatchValue(value=doc_type),
                        )
                    ]
                )

                # Temporarily add the filter to the search_kwargs of the ParentDocumentRetriever.
                # This is the key change: we modify the existing retriever instead of creating a new one.
                active_retriever.base_retriever.search_kwargs["filter"] = qdrant_filter
            else:
                logging.info("No doc_type filter applied. Using default compression retriever.")

            # The rest of the chain is built with the correctly configured retriever.
            retrieval_chain = create_retrieval_chain(active_retriever, self.question_answer_chain)

            logging.info("Invoking RAG chain...")
            invocation_start_time = time.time()
            result = await retrieval_chain.ainvoke({"input": question})
            invocation_end_time = time.time()
            logging.info(f" -> RAG chain invocation took {invocation_end_time - invocation_start_time:.2f} seconds.")

        finally:
            # IMPORTANT: Restore the original search_kwargs to ensure this query
            # doesn't affect the next one.
            self.retriever.base_retriever.search_kwargs = original_search_kwargs

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

            # --- FIX: Call add_documents on the underlying base_retriever ---
            # The ContextualCompressionRetriever is a wrapper; the actual document
            # handling is done by the ParentDocumentRetriever it contains.
            await self.retriever.base_retriever.add_documents([parent_doc], ids=None)

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