import re
from typing import Optional, List, Literal, Dict, Union, Any, Callable
from pydantic import BaseModel, Field, computed_field, model_validator, field_validator

from app.core.logging import get_logger
from app.model.enum.language import Languagecode

logger = get_logger()

class ChatRequest(BaseModel):
    uid: str = Field(...)
    state: int = Field(..., description="0: 시작, 1: 증상 부위 입력, 2: 설문 진행 중, 3: 요약, 치료 방법, 4: 종료")
    text: str = Field(...)
    lang: Optional[str] = Field(default="ko", description="ko: 한국어, en: 영어")


class ChatResponse(BaseModel):
    text: str
    screening: Optional[List[Dict[str, str | None]]] # 데이터 처리 고려한 Obejct List 타입으로 전달 
    treatment: Optional[List[str]]
    state: int = Field(default=0, description="0: 시작, 1: 증상 부위 입력, 2: 설문 진행 중, 3: 요약, 치료 방법, 4: 종료")
    treatment_time: Optional[int] = Field(default=None, description="치료 시간(분)")
    language: Optional[Languagecode] = Field(default=Languagecode.ko, description="ko: 한국어, en: 영어", exclude=True)

    class Config:
        from_attributes = True
        extra = 'allow'  # 정의되지 않은 속성 허용
    
    @classmethod
    def _clear_text(cls, _text: str) -> str:
        return _text.replace("lang:ko", "").replace("lang:en", "").replace("(", "").replace(")", "").strip()
    
    @field_validator("text", mode="before")
    @classmethod
    def check_text(
        cls, 
        value: str
    ) -> str:
        cls._original_text = value
        if "<question>" in value:
            tag_pattern = re.compile(r'<question>(.*?)</question>', re.DOTALL)
            match = tag_pattern.search(value)
            return cls._clear_text(_text=match.group(1).strip())
        elif "<treatment>" in value:
            tag_pattern = re.compile(r'<treatment>(.*?)</treatment>', re.DOTALL)
            match = tag_pattern.search(value)
            return cls._clear_text(_text=match.group(1).strip())
        else:
            return cls._clear_text(_text=value)

    @model_validator(mode="before")
    def check_screening(cls, values: Dict) -> Dict:
        logger.warning(f"check_screening: {values}")
        value = values.get("screening")
        text = values.get("text")
        language = values.get("language")  # Default to "ko" if language is not set
        if value is None:
            return values
        if language == "ko":
            result = {
                "증상": None,
                "증상 강도": None,
                "증상 부위": None,
                "지속 기간": None,
                "증상 유발요인": None,
                "하고 싶은 말": None
            }
        elif language == "en":
            result = {
                "Symptoms": None,
                "Severity": None,
                "Symptoms Area": None,
                "Duration": None,
                "Specific Situations": None,
                "Special Considerations": None
            }
        logger.warning(f"text: {text}")
        try:
            tag_pattern = re.compile(r'<screening>(.*?)</screening>', re.DOTALL)
            match = tag_pattern.search(value)
            if not match:
                match = tag_pattern.search(text)
            assert match, "No screening tag"

            content = match.group(1).strip()

            # 각 줄별로 분리합니다.
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            # 적어도 헤더, 구분선, 그리고 하나 이상의 데이터 행이 필요합니다.
            assert len(lines) > 2, "Not in screening format"

            # 첫 두 줄(헤더와 구분선)을 건너뜁니다.
            data_lines = lines[2:]

            for line in data_lines:
                # 파이프 문자 '|'로 시작 및 끝나는 경우 제거한 후 분할합니다.
                columns = [col.strip() for col in line.strip("|").split("|")]
                if len(columns) >= 2:
                    key, value = columns[0], columns[1]
                    result[key] = value.strip() if len(value.strip()) > 0 else None
        except AssertionError as e:
            logger.error(f"Reponse Parsing Error: {e}")
            result = result
        finally:
            values["screening"] = [{"label": k, "content": v} for k, v in result.items()]
            return values

    # @computed_field(description="0: 시작, 1: 증상 부위 입력, 2: 설문 진행 중, 3: 요약, 치료 방법, 4: 종료")
    # def state(
    #     self
    # ) -> int:
    #     tag_pattern = re.compile(r'<state>(.*?)</state>', re.DOTALL)
    #     match = tag_pattern.search(self._original_text)
    #     return 0 if not match else int(match.group(1).strip())


class RouterQuery(BaseModel):
    """Route query to destination."""
    destination: Literal["step1", "step2", "step3"]

class TreatmentQuery(BaseModel):
    """Route query to destination."""
    answers: List[Optional[str]] = Field(
        description="Selected treatments from give possible answers."
    )
    
    @model_validator(mode="before")
    def check_structured_output(cls, values: Dict) -> Dict:
        print("structed treatments selections : ", values)
        return values