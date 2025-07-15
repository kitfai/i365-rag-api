
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, EmailStr, computed_field
from typing import List, Optional, Literal
import os


from typing import Optional

class LLMQueryRequestDTO(BaseModel):
    """
    The request body for querying the RAG pipeline.
    """
    query: str
    doc_type: Optional[str] = None