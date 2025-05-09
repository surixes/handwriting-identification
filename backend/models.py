from pydantic import BaseModel

# Модели Pydantic
class UserCreate(BaseModel):
    full_name: str
    user_id: str
    notes: str = ""
    

class VerificationResult(BaseModel):
    name: str
    id: str
    match: float