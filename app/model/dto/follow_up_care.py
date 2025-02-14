from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    uid: str = Field(...)
    state: int = Field(..., description="0: 시작, 1: 증상 부위 입력, 2: 진행 중, 3: 요약, 4: 치료, 5: 종료")
    text: str = Field(...)
    lang: Optional[str] = Field(default="ko", description="ko: 한국어, en: 영어")

class ChatResponse(BaseModel):
    text: str
    state: int = Field(default=0, description="0: 시작, 1: 증상 부위 입력, 2: 진행 중, 3: 요약, 4: 치료, 5: 종료")

    class Config:
        from_attributes = True
