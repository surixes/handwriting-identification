from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Модели Pydantic
class UserCreate(BaseModel):
    full_name: str
    user_id: str
    notes: str = ""
    

class VerificationResult(BaseModel):
    name: str
    id: str
    match: float

class ApiKeyCreate(BaseModel):
    owner: str
    description: Optional[str] = None

class ApiKeyInfo(BaseModel):
    key: str
    owner: str
    created_at: datetime
    is_active: bool

class FeedbackRequest(BaseModel):
    name: str
    email: str
    subject: str
    message: str
    consent: bool