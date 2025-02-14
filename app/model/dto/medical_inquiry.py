from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    uid: str = Field(...)
    state: int = Field(..., description="0: 시작, 1: 증상 부위 입력, 2: 설문 진행 중, 3: 요약, 4: 치료 방법, 5: 종료")
    text: str = Field(...)
    lang: Optional[str] = Field(default="ko", description="ko: 한국어, en: 영어")

class ChatResponse(BaseModel):
    text: str
    state: int = Field(default=0, description="0: 시작, 1: 증상 부위 입력, 2: 설문 진행 중, 3: 요약, 4: 치료 방법, 5: 종료")

    class Config:
        from_attributes = True
        

class RouterQuery(BaseModel):
    """Route query to destination."""
    destination: Literal["step1", "step2", "step3"]

class TreatmentQuery(BaseModel):
    """Route query to destination."""
    answers: List[Optional[str]] = Field(
        default=None,
        description="Selected treatments. ex) ['치료1', '치료2', ...]"
    )
