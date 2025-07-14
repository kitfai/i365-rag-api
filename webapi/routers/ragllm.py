from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Union

# Note the relative imports for a file inside a package
from .. dto.request.requestschemas import LLMQueryRequestDTO
from .. dto.response.responseschemas import ResponseDTO,LLMQueryResponseDTO
# Create a new router object.
# We can add a prefix and tags for better organization and documentation.
router = APIRouter(
    prefix="/rag-llm",
    tags=["RAG-LLM"],
)

@router.post("/query-llm", response_model=ResponseDTO[LLMQueryResponseDTO])
def create_user(queryRequestDTO: LLMQueryRequestDTO, response_model=ResponseDTO[LLMQueryResponseDTO]):
    llm_payload = LLMQueryResponseDTO(llm_response="Hey, this is the response from the LLM!")


    response = ResponseDTO[LLMQueryResponseDTO](
        res=llm_payload
    )

    return response