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
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_core.runnables import Runnable
# FINAL FIX: Use modern, non-deprecated components and high-level constructors
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from unstructured.partition.auto import partition

# --- Qdrant & Configuration ---
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Define Document Types ---
DocType = Literal[
    "Invoice", "Interest Advice", "Billing", "Unknown"]


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
    LLM_MODEL: str = 'deepseek-r1:latest'
    LLM_CLASSIFIER_MODEL: str = 'deepseek-r1:latest'
    LLM_TRANSFORMER_MODEL: str = 'deepseek-r1:latest'
    LLM_TIMEOUT: int = 120
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = RagSettings()

# --- Setup Logging ---
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
    A high-performance RAG service using a Qdrant server, LLM-based classification,
    and content-aware chunking. Implemented as a Singleton.
    """
    _instance = None
    _initialized = False

    # Component chain for the RAG pipeline
    rag_chain: Runnable

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(QdrantRAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self, force_rebuild: bool = False):
        if self._initialized:
            return

        logging.info("--- Initializing Qdrant RAG Service (Singleton Instance) ---")

        self.qdrant_client = qdrant_client.QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.llm = OllamaLLM(model=settings.LLM_MODEL, timeout=settings.LLM_TIMEOUT)
        self.classifier_llm = OllamaLLM(model=settings.LLM_CLASSIFIER_MODEL, timeout=30)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.splitters: Dict[DocType, TextSplitter] = {
            "Invoice": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
            "Interest Advice": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
            "Billing": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
            "Unknown": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        }

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

        self._build_rag_chain()

        self._initialized = True
        logging.info("--- RAG Service is ready for queries. ---")

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
            # This is a robust way to get the embedding dimension
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

    async def process_new_documents(self):
        """Public method to run the async document processing task."""
        logging.info("Starting document ingestion process...")
        await self._process_and_upload_documents()
        logging.info("--- Document ingestion has finished. ---")

    def _build_rag_chain(self):
        """Builds the RAG chain using modern, high-level LangChain constructors."""

        # 1. Define the retriever
        # This retriever will automatically use the vectorstore to find relevant documents.
        retriever = self.vectorstore.as_retriever()

        # 2. Define the prompt template for the final answer generation (stuff chain)
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

        # 3. Create the "stuff" chain for answering the question
        # This chain takes a question and a list of documents and "stuffs" them into the prompt.
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        # 4. Create the final retrieval chain
        # This chain takes a user question, retrieves documents, and then passes them to the question_answer_chain.
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def _parse_llm_output(self, raw_answer: str) -> str:
        """
        Parses the raw output from the LLM to remove the <think> block and other
        internal thoughts, returning only the user-facing answer.
        """
        # The "## Summary Answer" is a reliable delimiter to find the start of the real answer.
        if "## Summary Answer" in raw_answer:
            # We split the string at the delimiter and prepend it to the result
            # to keep the header in the final output.
            clean_part = raw_answer.split("## Summary Answer", 1)[1]
            return "## Summary Answer" + clean_part.strip()

        # Fallback in case the model doesn't follow the format perfectly
        logging.warning("Could not find '## Summary Answer' delimiter in LLM output. Returning raw answer.")
        return raw_answer.strip()

    async def query(self, question: str, doc_type: Optional[str] = None) -> dict:
        """
        Performs a query using the constructed RAG chain.
        """
        if not question:
            return {"answer": "Please provide a question.", "context": []}

        # The doc_type filter is not used in this simplified chain, but we keep the parameter for future enhancements.
        logging.info(f"Service querying with: '{question}'")
        start_time = time.time()

        # Invoke the complete RAG chain with just the user's input
        result = await self.rag_chain.ainvoke({"input": question})

        end_time = time.time()
        logging.info(f"RAG chain invocation took {end_time - start_time:.2f} seconds.")


        # --- MODIFICATION: Parse the raw answer before returning it ---
        raw_answer = result.get("answer", "No answer could be generated.")
        clean_answer = self._parse_llm_output(raw_answer)
        '''
        return {
            "answer": result.get("answer", "No answer could be generated."),
            "context": result.get("context", [])
        }
        '''
        return {
            "answer": clean_answer,
            "context": result.get("context", [])
        }

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

        for doc_type_option in DocType.__args__:
            if doc_type_option.lower() == response_text.lower():
                return doc_type_option
        for doc_type_option in DocType.__args__:
            if doc_type_option.lower() in response_text.lower():
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
            logging.info(f"  -> Classified {pdf_file.name} as type: {doc_type}. Now splitting and storing.")
            doc_id = pdf_file.name.replace(" ", "_")
            parent_doc = LangchainDocument(
                page_content=markdown_content,
                metadata={"source": str(pdf_file), "doc_id": doc_id, "doc_type": doc_type}
            )
            splitter = self.splitters[doc_type]
            child_docs = splitter.split_documents([parent_doc])
            await self.vectorstore.aadd_documents(child_docs, ids=None)
            await self.docstore.amset([(doc_id, parent_doc)])
            logging.info(f"  -> DONE: Successfully processed and stored {pdf_file.name}.")
            return pdf_file.name
        except Exception as e:
            logging.error(f"  -> FAILED to process {pdf_file.name} after content extraction: {e}", exc_info=True)
            return None

    async def _process_and_upload_documents(self):
        logging.info("Step 1/3: Checking for already indexed files in Qdrant...")
        try:
            points, _ = self.qdrant_client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                limit=10000, with_payload=["metadata.source"]
            )
            indexed_files = {Path(p.payload['metadata']['source']).name for p in points if p.payload}
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