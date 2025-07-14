
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, EmailStr, computed_field
from typing import List, Optional, Literal
import os

class LLMQueryRequestDTO(BaseModel):
    query:str