import re
from typing import Optional, List, Literal, Dict, Union
from pydantic import BaseModel, Field, computed_field, model_validator, field_validator


class ChatRequest(BaseModel):
    uid: str = Field(...)
    state: int = Field(..., description="0: 시작, 1: 증상 부위 입력, 2: 설문 진행 중, 3: 요약, 4: 치료 방법, 5: 종료")
    text: str = Field(...)
    lang: Optional[str] = Field(default="ko", description="ko: 한국어, en: 영어")

class ChatResponse(BaseModel):
    text: str
    screening: Optional[List[Dict[str,str]]] # 데이터 처리 고려한 Obejct List 타입으로 전달 
    treatment: Optional[List[str]]

    class Config:
        from_attributes = True
    
    @field_validator("text", mode="before")
    @classmethod
    def check_text(
        cls, 
        value: str
    ) -> str:
        cls._original_text = value
        if "<treatment>" in value:
            tag_pattern = re.compile(r'<treatment>(.*?)</treatment>', re.DOTALL)
            match = tag_pattern.search(value)
            return match.group(1).strip()
        elif "<question>" in value:
            tag_pattern = re.compile(r'<question>(.*?)</question>', re.DOTALL)
            match = tag_pattern.search(value)
            return match.group(1).strip()
        else:
            return value

    @field_validator("screening", mode="before")
    @classmethod
    def check_screening(
        cls, 
        value: str
    ) -> str:
        if value is None:
            return value
        else:
            result = {
                "증상": None, 
                "증상 강도": None, 
                "증상 부위": None, 
                "지속 기간": None, 
                "증상 유발요인": None, 
                "하고 싶은 말": None
            }
            try:
                tag_pattern = re.compile(r'<screening>(.*?)</screening>', re.DOTALL)
                match = tag_pattern.search(value)
                if not match:
                    assert False, "No screening tag"

                content = match.group(1).strip()

                # 각 줄별로 분리합니다.
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                if len(lines) < 3:
                    # 적어도 헤더, 구분선, 그리고 하나 이상의 데이터 행이 필요합니다.
                    assert False, "Not in screening format"

                # 첫 두 줄(헤더와 구분선)을 건너뜁니다.
                data_lines = lines[2:]

                for line in data_lines:
                    # 파이프 문자 '|'로 시작 및 끝나는 경우 제거한 후 분할합니다.
                    columns = [col.strip() for col in line.strip("|").split("|")]
                    if len(columns) >= 2:
                        key, value = columns[0], columns[1]
                        result[key] = value
            except AssertionError as e:
                result = result
            finally:
                return [{"label": k, "content": v} for k, v in result.items()]

    @computed_field(description="0: 시작, 1: 증상 부위 입력, 2: 설문 진행 중, 3: 요약, 4: 치료 방법, 5: 종료")
    def state(
        self
    ) -> int:
        tag_pattern = re.compile(r'<state>(.*?)</state>', re.DOTALL)
        match = tag_pattern.search(self._original_text)
        return 0 if not match else int(match.group(1).strip())


class RouterQuery(BaseModel):
    """Route query to destination."""
    destination: Literal["step1", "step2", "step3"]

class TreatmentQuery(BaseModel):
    """Route query to destination."""
    answers: List[Optional[str]] = Field(
        default=None,
        description="Selected treatments. ex) ['치료1', '치료2', ...]"
    )
