from typing import List, Dict
import re
from operator import itemgetter
import time
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable, RunnableParallel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.schema import StrOutputParser

from app.core.langchain_module.llm import DDG_LLM, get_llm
from app.model.dto.medical_inquiry import TreatmentQuery


class EntityChain(Runnable):
    def __init__(self, system_prompt: str):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("user", "Utterance: {question}")
        ])
        self.chain = self.prompt | self.llm

    def intent_parser(self, text)->dict:
        result = {
            "증상": None, 
            "증상 강도": None, 
            "증상 부위": None, 
            "지속 기간": None, 
            "증상 유발요인": None, 
            "하고 싶은 말": None
        }
        tag_pattern = re.compile(r'<screening>(.*?)</screening>', re.DOTALL)
        match = tag_pattern.search(text)
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
    
    def invoke(self, input, config, **kwargs):
        # 입력 데이터를 처리하는 로직 구현
        start = time.time()
        history = input.get("chat_history", [])
        question = input.get("question", "")
        intent = self.chain.invoke({
            "history": history,
            "question": question
        })
        print(f"Elapsed time: {time.time() - start}")
        return {
            "result": f"{' '.join([his.content for his in history if his.type=='human'])} {question}", 
            "intent": intent,
            "question": question, 
            "history": history,
            "parsed_intent": self.intent_parser(intent.content),
            "language": input.get("language", "ko")
        }

    # 필요한 경우 batch, stream 등의 메서드도 구현 가능
    def batch(self, inputs):
        return [self.invoke(input) for input in inputs]


class TimerChain(Runnable):
    def treatment_rule(
        self, 
        treatments: List[Dict],
        pain: str
    ) -> int:
        def time_in_max(max_time: int)->int:
            processed_times = [int(a['time']) for a in treatments.values()]
            return {
                "cal_info": [t for t in processed_times],
                "total": min(sum(processed_times), max_time)
            }
        if pain is None:
            pain = "0"
        pain = int(pain[-1] if "-" in pain or "~" in pain else pain)

        if len(treatments) > 1:
            for treat in treatments:
                treatments[treat]['time'] = int(treatments[treat]['time'].replace("분", ""))
                if treatments[treat]["time"] <= 25:
                    treatments[treat]["time"] = 0
                if "스케일링" in treatments and "잇몸" in treat:
                    if pain < 7:
                        del treatments[treat] 
                    else:
                        del treatments["스케일링"]

            is_treat = [t for t in treatments if "치료" in t]
            is_couns = [t for t in treatments if "상담" in t]
            
            if len(is_couns) > 1: # 상담이 2개 이상인 경우
                return {
                    "treatments": [t for t in treatments],
                    **time_in_max(55 if any([t for t in treatments if "교정" in t]) else 40)
                }
            elif any(is_treat) and any(is_couns): # 치료와 상담이 모두 있는 경우
                for t in treatments:
                    if "상담" in t:
                        treatments[t]["time"] = 40 if "교정" in t else 25
                return {
                    "treatments": [t for t in treatments],
                    **time_in_max(100)
                }
                
            elif any(is_treat): # 치료가 포함된 경우
                if any([t for t in treatments if "신경" in t]): # 신경치료가 포함된 경우
                    result = [t for t in treatments if "신경" in t] # 신경치료만 계산
                    is_re = [r for r in result if "재신경" in r] # 재신경 치료인 경우 재신경 치료 우선
                    result = is_re.pop() if len(is_re) > 0 else result.pop()
                    return {
                        "treatments": [result],
                        "cal_info": treatments[result]["time"],
                        "total": treatments[result]["time"]
                    }
                else:
                    return {
                        "treatments": [t for t in treatments],
                        **time_in_max(100)
                    }
        else:
            for treat in treatments:
                treatments[treat]['time'] = int(treatments[treat]['time'].replace("분", ""))
            return {
                "treatments": [t for t in treatments],
                **time_in_max(1000)
            }

    def invoke(self, input, config, **kwargs):
        # 입력 데이터를 처리하는 로직 구현
        answers = input.get("answers", [])
        raw_context = input.get("raw_context", [])
        intents = input.get("parsed_intent", {})
        treatment_data = {}
        for treat in raw_context:
            print(treat)
            if any([a for a in answers if a in treat.metadata["치료"]]):
                treatment_data[treat.metadata["치료"]] = {
                    "time": treat.metadata["소요 시간"],
                    "diagnosis": treat.metadata["증상"]
                    }
        treatment_rule = self.treatment_rule(treatments=treatment_data, pain=intents["증상 강도"])
        treatment_message = ", ".join(treatment_rule["treatments"])
        treatment_time_message = f'{"+".join(f"{i}분" for i in treatment_rule["cal_info"])}={treatment_rule["total"]}분'
        input.update({
            "context": "Response Guide\n"
                      f"예상되는 진료는 {treatment_message} 이며, 진료 시간은 {treatment_time_message} 으로 예상됩니다.\n"
                       "*위 결과는 증상에 따라 예상되는 진료 및 요약이며 의학적인 진단이 아닙니다. "
                       "요약 시 누락이나 오역이 있을 수 있으며, "
                       "실제 치료 내용과 시간은 방사선 촬영 및 의료진의 진료에 따라 달라질 수 있습니다."})
        # print(input["context"])
        return {
            **input
        }

    # 필요한 경우 batch, stream 등의 메서드도 구현 가능
    def batch(self, inputs):
        return [self.invoke(input) for input in inputs]


class StepDispatcher(Runnable):
    """
    destination 값("step1", "step2", "step3")에 따라 해당 체인을 선택하여 실행하는 Runnable 클래스입니다.
    """
    def __init__(self, system_prompt: str, timer_prompt: str):
        self.system_prompt = system_prompt
        self.timer_prompt = timer_prompt
        self.llm = get_llm()  # get_llm()를 통해 LLM 인스턴스 가져옴

        # step1: 문진 진행
        self.chain_step1 = (
            RunnableParallel({
                "text": ChatPromptTemplate.from_messages([
                             SystemMessage(content=self.system_prompt),
                             MessagesPlaceholder("history"),
                             ("human", "Contexts:\n{context}\n\n"
                                       "Screened Intents:\n{intent}\n"
                                       "Utterance: {question}"
                                       "(이 발화에 대한 응답은 해당하는 언어로 해주세요: {language})\n"
                                       "Processing State: step1")
                        ])
                        | self.llm
                        | StrOutputParser(),
                "screening": itemgetter("intent")
                             | RunnableLambda(lambda x: x.content)
            })
        )

        # step2: 치료 방법 제시
        self.chain_step2 = (
            RunnablePassthrough.assign(
                answers=ChatPromptTemplate.from_messages([  
                        # 필요한 치료 방법만 선택하는 프롬프트
                        SystemMessage(
                            content="Contexts를 참고하여 현재 가장 필요한 치료 방법을 선택하세요. "
                                    "서로 다른 치료 방법이 적용되야 하는 경우에만 다중 선택 가능합니다. "
                                    "같은 치료 방법이라면 더 적절한 것 하나만 선택해야합니다."),
                        MessagesPlaceholder("history"),
                        ("human", "Contexts:\n{context}\n\n"
                                  "Screened Intents:\n{intent}\n"
                                  "Utterance: {question}\n"
                                  "Processing State: step2\n"
                                  "---\n"
                                  "answers:[{raw_treatment}]") ])
                        | self.llm.with_structured_output(TreatmentQuery)
                        | RunnableLambda(lambda x: list(set(x.answers))))
            | RunnablePassthrough.assign(
                text=RunnablePassthrough()
                     | TimerChain() # 시간 계산 chain
                     | ChatPromptTemplate.from_messages([
                        SystemMessage(content=self.system_prompt),
                        MessagesPlaceholder("history"),
                        ("human", "Contexts:\n{context}\n\n"
                                  "Screened Intents:\n{intent}\n"
                                  "Utterance: {question}"
                                  "(이 발화에 대한 응답은 해당하는 언어로 해주세요: {language})\n"
                                  "Processing State: step2\n\n")])
                     | self.llm
                     | StrOutputParser(),
                screening=itemgetter("intent")
                          | RunnableLambda(lambda x: x.content))
            |{
                "text" : itemgetter("text"),
                "screening": itemgetter("screening"),
                "treatment": itemgetter("answers")
            }
        )

        # step3: 예상 시간 계산
        self.chain_step3 = (
            RunnableParallel({
                "text": ChatPromptTemplate.from_messages([
                             SystemMessage(content=self.system_prompt),
                             MessagesPlaceholder("history"),
                             ("human", "Contexts:\n{context}\n\n"
                                       "Screened Intents:\n{intent}\n"
                                       "Utterance: {question}\n"
                                       "Output Language: {language}\n"
                                       "Processing State: step2")])
                        | self.llm
                        | StrOutputParser(),
                "screening": itemgetter("intent")
                             | RunnableLambda(lambda x: x.content),
                "treatment": itemgetter("raw_treatment")
            })
        )

    def invoke(self, input: dict, config: dict = None, **kwargs):
        destination = input.get("destination")
        print(f"route to {destination} chain")
        start = time.time()
        match destination:
            case "step1":
                result = self.chain_step1.invoke(input, config=config, **kwargs)
            case "step2":
                result = self.chain_step2.invoke(input, config=config, **kwargs)
            case "step3":
                result = self.chain_step3.invoke(input, config=config, **kwargs)
            case _:
                raise ValueError(f"Invalid destination: {destination}")
        print(f"{destination} chain Elapsed time: {time.time() - start}")
        return result

    def batch(self, inputs: list, config: dict = None, **kwargs):
        return [self.invoke(single_input, config=config, **kwargs) for single_input in inputs]
