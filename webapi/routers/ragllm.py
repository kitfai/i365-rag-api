# D:/infra365/codes/rag-git/webapi/routers/ragllm.py

from fastapi import APIRouter, Depends
# Note the relative imports for a file inside a package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.service.qdrant_service import QdrantRAGService
from ..dto.request.requestschemas import LLMQueryRequestDTO
from ..dto.response.responseschemas import ResponseDTO, LLMQueryResponseDTO, SourceDocumentDTO

# Create a new router object.
router = APIRouter(
    prefix="/rag-llm",
    tags=["RAG-LLM"],
)

# --- Singleton Service Instance ---
rag_service = QdrantRAGService()

def get_rag_service():
    """Dependency to provide the singleton RAG service instance."""
    return rag_service

# WORLD-CLASS FIX: Make the endpoint asynchronous
@router.post("/query-llm", response_model=ResponseDTO[LLMQueryResponseDTO])
async def query_llm(
    query_request: LLMQueryRequestDTO,
    service: QdrantRAGService = Depends(get_rag_service)
):
    """
    Accepts a query and returns a structured response from the RAG pipeline.
    An optional 'doc_type' can be provided to filter the search.
    """
    # Use await to call the asynchronous service method
    result = await service.query(
        question=query_request.query,
        doc_type=query_request.doc_type
    )

    source_docs = []
    if "context" in result and result["context"]:
        for doc in result["context"]:
            source_docs.append(
                SourceDocumentDTO(
                    source=doc.metadata.get('source', 'Unknown source'),
                    page_content=doc.page_content
                )
            )

    llm_payload = LLMQueryResponseDTO(
        answer=result.get("answer", "No answer could be generated."),
        source_documents=source_docs
    )

    response = ResponseDTO[LLMQueryResponseDTO](
        data=llm_payload
    )

    return response
