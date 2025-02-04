from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    uid: str = Field(...)
    state: int = Field(...)
    text: str = Field(...)

class ChatResponse(BaseModel):
    text: str
    progress:str = Field(default="chat")

    class Config:
        from_attributes = True
