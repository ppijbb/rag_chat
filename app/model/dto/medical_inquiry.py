from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    uid: str = Field(...)
    state: int = Field(..., description="0: start, 1: in progress, 2: summarize, 3:treatment, 4: end")
    text: str = Field(...)

class ChatResponse(BaseModel):
    text: str
    state: int = Field(default=0, description="0: start, 1: in progress, 2: summarize, 3:treatment, 4: end")

    class Config:
        from_attributes = True
