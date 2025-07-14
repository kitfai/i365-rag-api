from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, EmailStr, computed_field
from typing import List, Optional, Literal,TypeVar, Generic

import os
T = TypeVar('T')

class ErrorResponseDTO(BaseModel):
    errorMsg:str | None = None
    errorCode:int | None = None


class LLMQueryResponseDTO(BaseModel):
    llm_response: str | None = None


class ResponseDTO(ErrorResponseDTO, Generic[T]):
    res:T | None = None