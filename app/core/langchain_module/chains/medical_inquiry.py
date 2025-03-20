from typing import List, Dict
import re
from operator import itemgetter
from collections import ChainMap
import time
from logging import Logger

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable, RunnableParallel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.schema import StrOutputParser

from app.core.logging import get_logger
from app.core.langchain_module.llm import DDG_LLM, get_llm
from app.model.dto.medical_inquiry import TreatmentQuery, RouterQuery


class EntityChain(Runnable):
    def __init__(self, system_prompt: str):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("history"),
            ("user", "Utterance: {question}")
        ])
        self.chain = self.prompt | self.llm
        self.chain_logger = get_logger()

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
        history = input.get("chat_history", [])
        question = input.get("question", "")
        intent = self.chain.invoke({
            "history": history,
            "question": question
        })
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
    max_time:int = 90
    chain_logger: Logger = get_logger()

    def treatment_rule(
        self, 
        treatments: List[Dict],
        pain: str
    ) -> int:
        def time_in_max(max_time: int) -> int:
            processed_times = [int(a['time']) for a in treatments.values()]
            return {
                "cal_info": [t for t in processed_times],
                "total": min(sum(processed_times), max_time)
            }
        
        if pain is None:
            pain = "0"
        pain = int(pain[-1] if "-" in pain or "~" in pain else pain)
        self.chain_logger.debug(f"Pain level: {pain}")

        if len(treatments) > 1:
            self.chain_logger.debug(f"Multiple treatments found: {treatments}")
            for treat in treatments:
                if isinstance(treatments[treat]['time'], str):
                    treatments[treat]['time'] = int(treatments[treat]['time'].replace("분", ""))
                if treatments[treat]["time"] <= 25:
                    treatments[treat]["time"] = 0
                if "스케일링" in treatments and "잇몸" in treat:
                    if pain < 7:
                        del treatments[treat]
                        self.chain_loggerdebugo(f"Deleted treatment due to low pain: {treat}")
                    else:
                        del treatments["스케일링"]
                        self.chain_logger.debug("Deleted 스케일링 due to high pain")

            is_treat = [t for t in treatments if "치료" in t]
            is_couns = [t for t in treatments if "상담" in t]
            self.chain_logger.debug(f"Treatments: {is_treat}, Consultations: {is_couns}")
            
            if len(is_couns) > 1:  # 상담이 2개 이상인 경우
                self.chain_logger.debug("Multiple consultations found")
                return {
                    "treatments": [t for t in treatments],
                    **time_in_max(55 if any([t for t in treatments if "교정" in t]) else 40)
                }
            elif any(is_treat) and any(is_couns):  # 치료와 상담이 모두 있는 경우
                self.chain_logger.debug("Both treatments and consultations found")
                for t in treatments:
                    if "상담" in t:
                        treatments[t]["time"] = 40 if "교정" in t else 25
                return {
                    "treatments": [t for t in treatments],
                    **time_in_max(self.max_time)
                }
                
            elif any(is_treat):  # 치료가 포함된 경우
                self.chain_logger.debug("Treatments found")
                if any([t for t in treatments if "신경" in t]):  # 신경치료가 포함된 경우
                    self.chain_logger.debug("Nerve treatment found")
                    result = [t for t in treatments if "신경" in t]  # 신경치료만 계산
                    is_re = [r for r in result if "재신경" in r]  # 재신경 치료인 경우 재신경 치료 우선
                    result = is_re.pop() if len(is_re) > 0 else result.pop()
                    self.chain_logger.debug(f"Selected treatment: {result}")
                    return {
                        "treatments": [result],
                        "cal_info": treatments[result]["time"],
                        "total": treatments[result]["time"]
                    }
                else:
                    return {
                        "treatments": [t for t in treatments],
                        **time_in_max(self.max_time)
                    }
            else:
                self.chain_logger.info("Only consultations found")
                self.chain_logger.info(f"Consultations: {treatments}")
                return {
                    "treatments": [t for t in treatments],
                    **time_in_max(self.max_time)
                }
        else:
            self.chain_logger.info("Single treatment found")
            for treat in treatments:
                if isinstance(treatments[treat]['time'], str):
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
        language = input.get("language", "ko")
        treatment_data = {}
        self.chain_logger.debug(answers)
        for treat in raw_context:
            self.chain_logger.debug(treat)
            if any([a for a in answers if a in treat.metadata[f"치료"]]):
                treatment_data[treat.metadata[f"치료"]] = {
                    "time": treat.metadata["소요 시간"],
                    "diagnosis": treat.metadata["증상"]
                    }
        treatment_rule = self.treatment_rule(treatments=treatment_data, pain=intents["증상 강도"])
        self.chain_logger.debug(treatment_rule)
        treatment_message = ", ".join(treatment_rule["treatments"])
        treatment_time_message = f'{"+".join(f"{i}분" for i in treatment_rule["cal_info"])}={treatment_rule["total"]}분'
        input.update({
            "context": "Response Guide\n"
                      f"예상되는 진료는 {treatment_message} 이며, 진료 시간은 {treatment_time_message} 으로 예상됩니다.\n",
            "treatment_time": treatment_rule["total"]})
        self.chain_logger.debug(input["context"])
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
    def __init__(self, system_prompt: str, timer_prompt: str, treatment_prompt: str):
        self.llm = get_llm()  # get_llm()를 통해 LLM 인스턴스 가져옴
        self.chain_logger = get_logger()

        # step1: 문진 진행
        self.chain_step1 = (
            RunnableParallel(
                text=ChatPromptTemplate.from_messages([
                        SystemMessage(content=system_prompt),
                        MessagesPlaceholder("history"),
                        ("human", "Contexts:\n{context}\n\n"
                                  "Screened Intents:\n{intent}\n"
                                  "Utterance: {question}\n"
                                  "Language: {language}\n"
                                  "Activated State: step1") ])
                    | self.llm
                    | StrOutputParser(),
                screening=itemgetter("intent")
                          | RunnableLambda(lambda x: x.content)
            )
        )

        # step2: 치료 방법 제시
        self.chain_step2 = (
            RunnablePassthrough.assign(
                answers=ChatPromptTemplate.from_messages([  
                    # 필요한 치료 방법만 선택하는 프롬프트
                    SystemMessage(content=treatment_prompt),
                    MessagesPlaceholder("history"),
                    ("human", "Contexts:\n{context}\n\n"
                              "Screened Intents:\n{intent}\n"
                              "Utterance: {question}\n"
                              "---\n"
                              "Possible Anwers:[{raw_treatment}]") ])
                    | self.llm.with_structured_output(TreatmentQuery)
                    | RunnableLambda(lambda x: [ a for a in set(x.answers)]),
                pain=RunnableLambda(lambda x: x["parsed_intent"]["증상 강도"]),
                language=RunnableLambda(lambda x: x["language"]))
            | TimerChain()
            | RunnableParallel(
                text=RunnablePassthrough()
                     | ChatPromptTemplate.from_messages([
                        SystemMessage(content=system_prompt),
                        MessagesPlaceholder("history"),
                        ("human", "Contexts:\n{context}\n\n"
                                  "Screened Intents:\n{intent}\n"
                                  "Utterance: {question}\n"
                                  "Language: {language}\n"
                                  "Activated State: step2\n\n")])
                     | self.llm
                     | StrOutputParser(),
                screening=itemgetter("intent")
                          | RunnableLambda(lambda x: x.content),
                treatment_time=itemgetter("treatment_time"),
                treatment= itemgetter("answers"))
        )

        # step3: 예상 시간 계산
        self.chain_step3 = (
            RunnableParallel(
                text=ChatPromptTemplate.from_messages([
                        SystemMessage(content=system_prompt),
                        MessagesPlaceholder("history"),
                        ("human", "Contexts:\n{context}\n\n"
                                  "Screened Intents:\n{intent}\n"
                                  "Utterance: {question}\n"
                                  "Language: {language}\n"
                                  "Activated State: step2")])
                     | self.llm
                     | StrOutputParser(),
                screening=itemgetter("intent")
                          | RunnableLambda(lambda x: x.content),
                treatment=itemgetter("raw_treatment")
            )
        )

    def invoke(self, input: dict, config: dict = None, **kwargs):
        destination = input.get("destination")
        self.chain_logger.info(f"route to {destination} chain")
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
        self.chain_logger.info(f"{destination} chain Elapsed time: {time.time() - start}")
        return result

    def batch(self, inputs: list, config: dict = None, **kwargs):
        return [self.invoke(single_input, config=config, **kwargs) for single_input in inputs]


class ServiceChain:
    """
    A class that creates and manages the RAG service chain for medical inquiries.
    Moved from MedicalInquiryService to improve modularity.
    """
    def __init__(self, 
                 retriever,
                 llm=None,
                 dental_section_list=None,
                 chain_logger=None):
        """
        Initialize the ServiceChain with required components.
        
        Args:
            retriever: The retriever to use for RAG
            llm: Language model to use, defaults to get_llm()
            dental_section_list: List of dental sections/areas
            process_context_func: Function to process context from retrieval
            chain_logger: Logger to use for timing information
        """
        self.llm = llm or get_llm()
        self.rag = retriever
        self.dental_section_list = dental_section_list or []
        self.chain_logger = chain_logger
    
    def _process_context(
        self, 
        step_output
    ) -> Dict[str, str]:
        result, category = [], []
        language = step_output["language"]
        self.chain_logger.warning(f"rag step output {step_output['rag']}")
        for doc in step_output['rag']:
            source_data = doc.page_content.strip()
            self.chain_logger.debug(language)
            doc.metadata["치료"] = doc.metadata.get(f"치료_{language}")
            self.chain_logger.debug(doc.metadata["치료"])
            metadata_text = "\n".join([
                f"{k}: {v}"
                for k, v in doc.metadata.items()
                if k not in ["_id", "_collection_name"]])
            if doc.metadata.get(f"치료") not in category:
                category.append(doc.metadata.get(f"치료"))
                result.append(f"Case {len(category)}\n"
                              f"유사 사례: {source_data}\n"
                              f"{metadata_text}")
        return {
            "context": "\n\n".join(result),
            "raw_context": step_output["rag"]
        }
   
    def build_chain(
        self,
        memory,
        language="ko"
    ) -> RunnablePassthrough:
        from datetime import datetime
        from app.util.time_func import format_datetime_with_ampm
        from app.model.dto.medical_inquiry import RouterQuery
        from app.core.prompts.medical_inquiry import (
            SYSTEM_PROMPT,STEP2_SYSTEM_PROMPT, ENTITY_PROMPT_KO, ENTITY_PROMPT_EN, TIMER_PROMPT, 
            STEP_CONTROL_PROMPT, TREATMENT_PROMPT)
        
        # Use provided templates or fall back to imported ones
        current_time = format_datetime_with_ampm(datetime.now())
        system_prompt_template = SYSTEM_PROMPT
        step2_system_prompt_template = STEP2_SYSTEM_PROMPT
        entity_prompt_ko = ENTITY_PROMPT_KO
        entity_prompt_en = ENTITY_PROMPT_EN
        timer_prompt_template = TIMER_PROMPT
        step_control_prompt_template = STEP_CONTROL_PROMPT
        treatment_prompt_template = TREATMENT_PROMPT
        
        # Format prompts with current time and dental sections
        
        system_prompt = system_prompt_template.format(
            current_time, 
            ", ".join(self.dental_section_list),
            language)
        step2_prompt = step2_system_prompt_template.format(
            current_time,
            language)
        entity_prompt = (entity_prompt_ko if language == "ko" else entity_prompt_en).format(current_time)
        timer_prompt = timer_prompt_template.format(current_time)
        step_prompt = step_control_prompt_template.format(current_time, ", ".join(self.dental_section_list))
        treatment_prompt = treatment_prompt_template.format(current_time)
        
        def timed_stage(stage_name, runnable):
            def timed_fn(*args):
                start = time.time()
                result = (runnable.invoke(input=args[0], config=None)
                            if hasattr(runnable, "invoke") else
                          runnable(input=args[0], config=None))
                elapsed = time.time() - start
                if self.chain_logger:
                    self.chain_logger.info(f"{stage_name} took {elapsed:.4f} seconds")
                return result
            return RunnableLambda(timed_fn)
        
        # Break down the pipeline into stages so we can time each one
        stage1 = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables)
                         | RunnableLambda(lambda x: x[memory.memory_key])
        )
        # stage2 = EntityChain(system_prompt=entity_prompt)
        stage2 = (RunnableParallel(
                    entity=EntityChain(system_prompt=entity_prompt),
                    context=RunnablePassthrough.assign(
                                rag=RunnableLambda(lambda x: f"{' '.join([his.content for his in x["chat_history"] if his.type=='human'])} {x['question']}")
                                    | self.rag, 
                                language=itemgetter("language"))
                            | self._process_context)
                  | RunnableLambda(lambda x: dict(ChainMap({"context":x["context"]}, x["entity"]))))
        stage3 = RunnableParallel(
            destination=ChatPromptTemplate.from_messages([
                            SystemMessage(content=step_prompt),
                            MessagesPlaceholder("history"),
                            ("human", "Screened Intents:\n"
                                      "{intent}\n"
                                      "Utterance: {question}")])
                        | self.llm.with_structured_output(RouterQuery)
                        | RunnableLambda(lambda x: x.destination),
            # context=RunnablePassthrough.assign(
            #             rag=itemgetter("result")
            #                 | self.rag, 
            #             language=itemgetter("language"))
            #         | self._process_context,
            context=itemgetter("context"),
            question=itemgetter("question"),
            intent=itemgetter("intent"),
            parsed_intent=itemgetter("parsed_intent"),
            history=itemgetter("history"),
            language=itemgetter("language")
        )
        stage4 = RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["context"]["context"]),
            raw_context=RunnableLambda(lambda x: x["context"]["raw_context"]),
            raw_treatment=RunnableLambda(lambda x: [data.metadata.get(f"치료") for data in x["context"]["raw_context"]])
        )
        stage5 = StepDispatcher(
            system_prompt=system_prompt,
            timer_prompt=timer_prompt,
            treatment_prompt=treatment_prompt
        )
        
        # Compose the pipeline while wrapping each stage with the timing wrapper
        timed_chain = (
            timed_stage("Stage 1: Load Memory and Chat History", stage1)
            | timed_stage("Stage 2: EntityChain", stage2)
            | timed_stage("Stage 3: RunnableParallel", stage3)
            | timed_stage("Stage 4: Context Assignment", stage4)
            | timed_stage("Stage 5: StepDispatcher", stage5)
        )
        return timed_chain
