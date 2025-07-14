from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Union

# Note the relative imports for a file inside a package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.service.qdrant_service import QdrantRAGService
from ..dto.request.requestschemas import LLMQueryRequestDTO
from ..dto.response.responseschemas import ResponseDTO, LLMQueryResponseDTO, SourceDocumentDTO
# Create a new router object.
# We can add a prefix and tags for better organization and documentation.
router = APIRouter(
    prefix="/rag-llm",
    tags=["RAG-LLM"],
)

# --- Singleton Service Instance ---
# This creates the service object once when the API starts.
# All API calls will reuse this single instance.
rag_service = QdrantRAGService()

def get_rag_service():
    """Dependency to provide the singleton RAG service instance."""
    return rag_service

@router.post("/query-llm", response_model=ResponseDTO[LLMQueryResponseDTO])
def query_llm(
    query_request: LLMQueryRequestDTO,
    service: QdrantRAGService = Depends(get_rag_service)
):
    """
    Accepts a query and returns a structured response from the RAG pipeline.
    """
    # 1. Use the injected service to perform the query
    result = service.query(question=query_request.query)

    # 2. Format the source documents for the API response
    source_docs = []
    if "context" in result and result["context"]:
        for doc in result["context"]:
            source_docs.append(
                SourceDocumentDTO(
                    source=doc.metadata.get('source', 'Unknown source'),
                    page_content=doc.page_content
                )
            )

    # 3. Build the final response payload
    llm_payload = LLMQueryResponseDTO(
        answer=result.get("answer", "No answer could be generated."),
        source_documents=source_docs
    )

    response = ResponseDTO[LLMQueryResponseDTO](
        data=llm_payload
    )

    return response