from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, EmailStr, computed_field
from typing import List, Optional, Literal,TypeVar, Generic

import os
T = TypeVar('T')

class ErrorResponseDTO(BaseModel):
    errorMsg:str | None = None
    errorCode:int | None = None


class SourceDocumentDTO(BaseModel):
    """Represents a source document used for the answer."""
    source: str
    page_content: str

class LLMQueryResponseDTO(BaseModel):
    """The specific data payload for an LLM query response."""
    answer: str
    source_documents: List[SourceDocumentDTO] = []

class ResponseDTO(ErrorResponseDTO, Generic[T]):
    """A generic, standardized response wrapper."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[T] = None