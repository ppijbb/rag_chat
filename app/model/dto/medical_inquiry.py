import re
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, computed_field, model_validator, field_validator

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
    
    @computed_field(description="아직 파악이 안된 경우 null로 반환.")
    def screening(self)->str:
        result = {
            "증상": None, 
            "증상 강도": None, 
            "증상 부위": None, 
            "증상 기간": None, 
            "증상 유발요인": None, 
            "하고싶은 말": None
        }
        tag_pattern = re.compile(r'<screening>(.*?)</screening>', re.DOTALL)
        match = tag_pattern.search(self.text)
        if not match:
            return result

        content = match.group(1).strip()

        # 각 줄별로 분리합니다.
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) < 3:
            # 적어도 헤더, 구분선, 그리고 하나 이상의 데이터 행이 필요합니다.
            return result

        # 첫 두 줄(헤더와 구분선)을 건너뜁니다.
        data_lines = lines[2:]

        for line in data_lines:
            # 파이프 문자 '|'로 시작 및 끝나는 경우 제거한 후 분할합니다.
            columns = [col.strip() for col in line.strip("|").split("|")]
            if len(columns) >= 2:
                key, value = columns[0], columns[1]
                result[key] = value
        return result
    
    @computed_field(description="아직 분석이 안된 경우 null로 반환.")
    def treatment(self)->str:
        tag_pattern = re.compile(r'<treatment>(.*?)</treatment>', re.DOTALL)
        match = tag_pattern.search(self.text)
        if not match:
            return None
        return match.group(1).strip()
    
    @field_validator("text")
    @classmethod
    def check_output(cls, value:str )->str:
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
    

class RouterQuery(BaseModel):
    """Route query to destination."""
    destination: Literal["step1", "step2", "step3"]

class TreatmentQuery(BaseModel):
    """Route query to destination."""
    answers: List[Optional[str]] = Field(
        default=None,
        description="Selected treatments. ex) ['치료1', '치료2', ...]"
    )
