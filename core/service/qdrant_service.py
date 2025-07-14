# D:/infra365/codes/rag-git/core/service/qdrant_service.py

import pickle
from pathlib import Path

# --- Core LangChain components ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
# Move configuration to a shared location or keep it here for the service
ROOT_DIR = Path(__file__).resolve().parents[2]  # Go up to the project root
PDFS_PATH = ROOT_DIR / "batch_process" / "pdf"
DB_PATH = ROOT_DIR / "batch_process" / "vectorstore_qdrant"
DOCSTORE_PATH = DB_PATH / "docstore"
QDRANT_COLLECTION_NAME = "rag_parent_documents"
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
LLM_MODEL = 'deepseek-r1:latest'


class QdrantRAGService:
    """
    A service class to encapsulate all RAG logic including indexing,
    retrieval, and querying.
    """

    def __init__(self, force_rebuild: bool = False):
        """
        Initializes the service, setting up the retriever and the RAG chain.
        This can be a long-running process if the DB needs to be built.
        """
        print("--- Initializing Qdrant RAG Service ---")
        self.retriever = self._get_or_create_retriever(force_rebuild)

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
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        print("--- RAG Service is ready. ---")

    def query(self, question: str) -> dict:
        """
        Performs a query against the RAG chain.
        """
        if not question:
            return {"answer": "Please provide a question.", "context": []}

        print(f"Service querying with: '{question}'")
        result = self.rag_chain.invoke({"input": question})
        return result

    def _process_pdf_to_markdown(self, pdf_path: Path) -> str | None:
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

    def _get_or_create_retriever(self, force_rebuild: bool) -> ParentDocumentRetriever:
        """
        The core logic from the original script to build/update the vector store
        and initialize the ParentDocumentRetriever.
        """
        # This is the exact same logic as your original get_retriever function,
        # just refactored as a method of this class.
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        qdrant_client_instance = qdrant_client.QdrantClient(path=str(DB_PATH))

        collections_response = qdrant_client_instance.get_collections()
        collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections_response.collections)

        pdfs_to_process = []
        if force_rebuild or not collection_exists:
            print("--- Building new Vector DB and Docstore ---")
            DB_PATH.mkdir(exist_ok=True)
            vector_size = len(embeddings.embed_query("test query"))
            qdrant_client_instance.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
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

        vectorstore = QdrantVectorStore(
            client=qdrant_client_instance,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )

        store = EncoderBackedStore(
            LocalFileStore(root_path=str(DOCSTORE_PATH)),
            lambda key: key, pickle.dumps, pickle.loads
        )

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            id_key="doc_id"
        )

        if not pdfs_to_process and not (force_rebuild or not collection_exists):
            return retriever

        all_parent_docs = []
        for pdf_file in pdfs_to_process:
            content = self._process_pdf_to_markdown(pdf_file)
            if content:
                doc_id = str(pdf_file.name)
                doc = Document(page_content=content, metadata={"source": str(pdf_file), "doc_id": doc_id})
                all_parent_docs.append(doc)

        if all_parent_docs:
            print(f"Adding {len(all_parent_docs)} documents to the retriever...")
            retriever.add_documents(all_parent_docs, ids=None)
            print("--- DB and Docstore update complete. ---")

        return retriever