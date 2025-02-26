from operator import itemgetter
from typing import List, Any, Dict, AsyncIterator
import datetime
import time

from ray import serve

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MultiQueryRetriever

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSerializable

from langchain_community.retrievers import BM25Retriever

from app.core.langchain_module.rag import VectorStore
from app.core.langchain_module.llm import DDG_LLM, get_llm
from app.util.time_func import format_datetime_with_ampm
from app.core.langchain_module.chains.medical_inquiry import EntityChain, StepDispatcher
from app.model.dto.medical_inquiry import RouterQuery
from app.core.prompts.medical_inquiry import SYSTEM_PROMPT, ENTITY_PROMPT_KO, ENTITY_PROMPT_EN, TIMER_PROMPT, MULTI_QUERY_PROMPT
from app.service._base import BaseService


@serve.deployment(
    placement_group_bundles=[{
        "CPU": 1.0, 
        }], 
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_ongoing_requests": 5,
    },
    placement_group_strategy="STRICT_PACK",
    max_ongoing_requests=10)
class MedicalInquiryService(BaseService):
    # dental_section_list = [
    #     "혀", "입천장", 
    #     "좌측 턱", "우측 턱", 
    #     "상악 좌측치", "하악 좌측치", 
    #     "상악 전치부", "하악 전치부", 
    #     "상악 우측치", "하악 우측치"]
    dental_section_list = [
        "혀", "입천장",
        "왼쪽 턱", "오른쪽 턱",
        "왼쪽 위", "왼쪽 아래",
        "위 앞니", "아래 앞니",
        "오른쪽 위", "오른쪽 아래"]
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.llm = kwargs.get("llm", get_llm())
        self.collection_name = kwargs.get("collection_name", "pre_screening")
        self.vectorstore = VectorStore(collection_name=self.collection_name)
        self.rag = self.get_adaptive_retriever(
            vectorstore=self.vectorstore.vectorstore,
            compressor=self.vectorstore.reranker,
            vector_docs=self.vectorstore.all_docs)
        self._memory_path = "qdrant_storage"
        
    async def inquiry_chat(
        self,
        text:str,
        language:str,
        state:int,
        memory_key:str="history"
    ) -> Dict[str, str]:
        start = time.time()
        rag_chain = self.get_service_chain(
            memory=ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(
                    messages=self._get_user_history(
                        memory_key=memory_key,
                        state=state)),
                memory_key=memory_key,
                return_messages=True),
            language=language
            )
        self.service_logger.info(f"chain init takes {time.time()-start}")
        start = time.time()
        result = await rag_chain.ainvoke({
                "question": text,
                "language": language
            })
        self.service_logger.info(f"chain run takes {time.time()-start}")
        await self._add_user_history( # 채팅 기록 저장
            memory_key=memory_key,
            state=state,
            data=[
                HumanMessage(content=text), 
                AIMessage(content=result["text"].strip())
            ])
        return result

    async def inquiry_stream(
        self,
        text:str,
        language:str,
        state:int,
        memory_key:str="history"
    ) -> AsyncIterator:
        rag_chain = self.get_service_chain(
            memory=ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(
                    messages=self._get_user_history(
                        memory_key=memory_key,
                        state=state)),
                return_messages=True,
                memory_key=memory_key),
            language=language
            )
        return rag_chain.astream({
                "question": text,
                "language": language
            })

    def get_adaptive_retriever(
        self, 
        vectorstore: VectorStore, 
        compressor: CrossEncoderReranker,
        vector_docs: List[Document] = []
    ) -> ContextualCompressionRetriever:
        # Base vector retriever
        k = 5
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",  # similarity, similarity_score_threshold, mmr
            search_kwargs={
                "k": k,
                "score_threshold": 0.6,
                # "fetch_k": 20,
                # "lambda_mult": 0.7
            }
        )
        
        # Multi-query retriever
        # multi_query_retriever = vector_retriever.vectorstore.as_retriever()
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_retriever,
            llm=self.llm,
            # parser_key="lines", # parser_key는 더 이상 사용하지 않음
            include_original=True,
            prompt=PromptTemplate(
                template=MULTI_QUERY_PROMPT
            )
        )

        # 문서가 있을 때만 BM25와 Ensemble 사용
        if vector_docs:
            # BM25 retriever 설정
            bm25_retriever = BM25Retriever.from_documents(
                documents=vector_docs,
                bm25_params={
                    "b": 0.75,
                    "k1": 1.2                
                })
            bm25_retriever.k = k  # 검색 결과 수 설정

            # Ensemble retriever (Vector + BM25)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[multi_query_retriever, bm25_retriever],
                weights=[0.6, 0.4],  # 벡터 검색에 더 높은 가중치
                c=15
            )
            base_retriever = ensemble_retriever
        else:
            # 문서가 없으면 multi-query retriever만 사용
            print("데이터베이스가 비어있어 벡터 검색만 수행합니다.")
            base_retriever = multi_query_retriever

        # Compression/Reranking retriever 적용
        # compressor = LLMChainExtractor.from_llm(
        #     llm=get_llm(),
        #     prompt=PromptTemplate(
        #     template=(
        #         "Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. "
        #         "If none of the context is relevant return NO_OUTPUT.\n\n"
        #         "Remember, *DO NOT* edit the extracted parts of the context.\n\n"
        #         "> Question: {question}\n"
        #         "> Context:\n"
        #         ">>>\n"
        #         "{context}\n"
        #         ">>>\n"
        #         "Extracted relevant parts:"),
        #     input_variables=["question", "context"],
        #     output_parser=NoOutputParser()
        #     )
        # )
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return final_retriever
    
    def _process_context(
        self, 
        step_output
    ) -> Dict[str, str]:
        result, category = [], []
        language = step_output["language"]
        self.service_logger.warning(f"rag step output {step_output['rag']}")
        for doc in step_output['rag']:
            source_data = doc.page_content.strip()
            doc.metadata["치료"] = doc.metadata.get(f"치료_{language}")
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

    # Initialize RAG chain
    def get_service_chain(
        self,
        memory: ConversationBufferMemory,
        language: str,
    ) -> RunnableSerializable:

        system_prompt: str = SYSTEM_PROMPT.format(
            format_datetime_with_ampm(datetime.datetime.now()), # 현재 시각
            ", ".join(self.dental_section_list),
            language) # 통증 부위 목록
        entity_prompt: str = (ENTITY_PROMPT_KO if language == "ko" else ENTITY_PROMPT_EN).format(
            format_datetime_with_ampm(datetime.datetime.now()), # 현재 시각
            )
        timer_prompt: str = TIMER_PROMPT.format(
            format_datetime_with_ampm(datetime.datetime.now())
            )
        # IntentChain과 RunnableParallel 이후 결과를 router_chain을 통해 분기시킵니다.
        # return (RunnablePassthrough.assign(
        #              chat_history=RunnableLambda(memory.load_memory_variables)
        #                           | RunnableLambda(lambda x: x[memory.memory_key]))
        #         | EntityChain(system_prompt=entity_prompt)
        #         | RunnableParallel(
        #                 destination=ChatPromptTemplate.from_messages([
        #                                 SystemMessage(content=system_prompt + "\n최종적으로 다음으로 실행해야 하는 Step을 결정하세요."),
        #                                 MessagesPlaceholder("history"),
        #                                 ("human", "Screened Intents:\n{intent}\nUtterance: {question}")])
        #                             | self.llm.with_structured_output(RouterQuery)
        #                             | RunnableLambda(lambda x: x.destination),
        #                 context={
        #                     "rag": itemgetter("result")
        #                            | self.rag, 
        #                     "language": itemgetter("language")
        #                     }
        #                     | self._process_context,
        #                 question=itemgetter("question"),
        #                 intent=itemgetter("intent"),
        #                 parsed_intent=itemgetter("parsed_intent"),
        #                 history=itemgetter("history"),
        #                 language=itemgetter("language")
        #             )
        #         | RunnablePassthrough.assign(
        #             context=RunnableLambda(lambda x: x["context"]["context"]),
        #             raw_context=RunnableLambda(lambda x: x["context"]["raw_context"]),
        #             raw_treatment=RunnableLambda(lambda x: [data.metadata.get("치료") for data in x["context"]["raw_context"]]))
        #         | StepDispatcher(
        #             system_prompt=system_prompt, 
        #             timer_prompt=timer_prompt)
        #     )

        def timed_stage(stage_name: str, runnable):
            def timed_fn(*args):
                import time
                start = time.time()
                # Execute the given runnable (using .invoke if available, otherwise call it)
                result = runnable.invoke(input=args[0], config=None) if hasattr(runnable, "invoke") else runnable(input=args[0], config=None)
                elapsed = time.time() - start
                self.service_logger.info(f"{stage_name} took {elapsed:.4f} seconds")
                return result
            return RunnableLambda(timed_fn)
        
        # Break down the original pipeline into stages so we can time each one.
        stage1 = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables)
                         | RunnableLambda(lambda x: x[memory.memory_key])
        )
        stage2 = EntityChain(system_prompt=entity_prompt)
        stage3 = RunnableParallel(
            destination=ChatPromptTemplate.from_messages([
                            SystemMessage(content=system_prompt + "\n최종적으로 다음으로 실행해야 하는 Step을 결정하세요."),
                            MessagesPlaceholder("history"),
                            ("human", "Screened Intents:\n{intent}\nUtterance: {question}")])
                        | self.llm.with_structured_output(RouterQuery)
                        | RunnableLambda(lambda x: x.destination),
            context=RunnablePassthrough.assign(
                        rag=itemgetter("result")
                            | self.rag, 
                        language=itemgetter("language"))
                    | self._process_context,
            question=itemgetter("question"),
            intent=itemgetter("intent"),
            parsed_intent=itemgetter("parsed_intent"),
            history=itemgetter("history"),
            language=itemgetter("language")
        )
        stage4 = RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["context"]["context"]),
            raw_context=RunnableLambda(lambda x: x["context"]["raw_context"]),
            raw_treatment=RunnableLambda(lambda x: [data.metadata.get("치료") for data in x["context"]["raw_context"]])
        )
        stage5 = StepDispatcher(
            system_prompt=system_prompt,
            timer_prompt=timer_prompt
        )
        
        # Compose the pipeline while wrapping each stage with the timing wrapper.
        timed_chain = (
            timed_stage("Stage 1: Load Memory and Chat History", stage1)
            | timed_stage("Stage 2: EntityChain", stage2)
            | timed_stage("Stage 3: RunnableParallel", stage3)
            | timed_stage("Stage 4: Context Assignment", stage4)
            | timed_stage("Stage 5: StepDispatcher", stage5)
        )
        return timed_chain
