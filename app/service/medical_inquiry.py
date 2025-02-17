from abc import ABC
from operator import itemgetter
from typing import List,  Any
import datetime
import shelve

from ray import serve

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MultiQueryRetriever

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSerializable

from langchain_community.retrievers import BM25Retriever

from app.core.logging import get_logger
from app.core.langchain_module.rag import VectorStore
from app.core.langchain_module.llm import DDG_LLM, get_llm
from app.util.time_func import format_datetime_with_ampm
from app.core.langchain_module.chains.medical_inquiry import EntityChain, StepDispatcher
from app.model.dto.medical_inquiry import RouterQuery
from app.core.prompts.medical_inquiry import SYSTEM_PROMPT, ENTITY_PROMPT, TIMER_PROMPT, MULTI_QUERY_PROMPT


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
class MedicalInquiryService:
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
        self.llm = kwargs.get("llm", DDG_LLM())
        self.collection_name = kwargs.get("collection_name", "pre_screening")
        self.vectorstore = VectorStore(collection_name=self.collection_name)
        self._memory_path = "qdrant_storage"
        self.logger = get_logger()

    def _get_user_history(self, memory_key:str):
        with shelve.open(f'{self._memory_path}/{memory_key}') as db:
            history = db.get("history", [])
            self.logger.warning(db)
            self.logger.warning(history)
        return history
    
    def _add_user_history(self, memory_key:str, data: Any):
        with shelve.open(f'{self._memory_path}/{memory_key}') as db:
            history = db.get("history", [])
            if isinstance(data, (tuple, list)):
                for d in data:
                    self.logger.warning(d)
                    history.append(d)
            else:
                history.append(data)
        self.logger.warning(history)
        
    async def inquiry_chat(
        self,
        text:str,
        memory_key: str = "history"
    ):
        self.logger.warning(text)
        rag_chain = self.get_rag_chain(
            vectorstore=self.vectorstore,
            memory=ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(
                    messages=self._get_user_history(memory_key)),
                return_messages=True,
                memory_key=memory_key)
            )
        result = await rag_chain.ainvoke({"input": {"question": text}})
        self._add_user_history(memory_key, [("user", text), ("ai", result)])
        return result

    async def inquiry_stream(
        self,
        text:str,
        memory_key: str = "history"
    ):
        rag_chain = self.get_rag_chain(
            vectorstore=self.vectorstore,
            memory=ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(
                    messages=self._get_user_history(memory_key)),
                return_messages=True,
                memory_key=memory_key)
            )
        return rag_chain.astream({"input": {"question": text}})

    # Adaptive RAG components
    def generate_queries(self, question: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
                # Start of Selection
                ("system", "제공된 질문에 대해 관련 컨텍스트를 검색하기 위해 3가지 다른 버전의 질문을 생성하세요. 다양하게 만드세요."),
                ("user", "{question}")
            ])
        chain = prompt | self.llm | LineListOutputParser()
        return chain.invoke({"question": question})

    def get_adaptive_retriever(self, vectorstore: VectorStore, compressor: CrossEncoderReranker):
        # Base vector retriever
        k = 5
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",  # similarity, similarity_score_threshold, mmr
            search_kwargs={
                "k": k,
                "score_threshold": 0.7,
                # "fetch_k": 20,
                # "lambda_mult": 0.7
            }
        )
        
        # Multi-query retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_retriever,
            llm=self.llm,
            # parser_key="lines", # parser_key는 더 이상 사용하지 않음
            include_original=True,
            prompt=PromptTemplate(
                template=MULTI_QUERY_PROMPT
            )
        )

        # BM25 retriever 초기화
        # vectorstore에서 모든 문서 가져오기
        all_docs = []
        try:
            results = vectorstore.client.scroll(
                collection_name=vectorstore.collection_name,
                limit=1000  # 적절한 수로 조정
            )[0]
            for result in results:
                if result.payload.get("page_content"):  # 유효한 문서만 추가
                    all_docs.append(
                        Document(
                            page_content=result.payload.get("page_content", ""),
                            metadata=result.payload.get("metadata", {})
                        )
                    )
        except Exception as e:
            print(f"BM25 초기화 중 오류 발생: {e}")
            all_docs = []

        # 문서가 있을 때만 BM25와 Ensemble 사용
        if all_docs:
            # BM25 retriever 설정
            bm25_retriever = BM25Retriever.from_documents(
                documents=all_docs,
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
    
    def _process_context(self, step_output):
        result, category = [], []
        for doc in step_output:
            source_data = doc.page_content.strip()
            metadata_text = "\n".join([f"{k}: {v}"
                                    for k, v in doc.metadata.items()
                                    if k not in ["_id", "_collection_name"]])
            if doc.metadata.get("치료") not in category:
                category.append(doc.metadata.get("치료"))
                result.append(f"Case {len(category)}\n"
                            f"유사 사례: {source_data}\n"
                            f"{metadata_text}")
        return {"context":"\n\n".join(result), "raw_context": step_output}

    # Initialize RAG chain
    def get_rag_chain(
        self,
        vectorstore: VectorStore, 
        memory: ConversationBufferMemory
        )-> RunnableSerializable:

        system_prompt: str = SYSTEM_PROMPT.format(
            format_datetime_with_ampm(datetime.datetime.now()), # 현재 시각
            ", ".join(self.dental_section_list)) # 통증 부위 목록
        entity_prompt: str = ENTITY_PROMPT.format(
            format_datetime_with_ampm(datetime.datetime.now()), # 현재 시각
            )
        timer_prompt: str = TIMER_PROMPT.format(
            format_datetime_with_ampm(datetime.datetime.now())
            )
        # IntentChain과 RunnableParallel 이후 결과를 router_chain을 통해 분기시킵니다.
        return ({"chat_history": RunnablePassthrough.assign(
                                    history=RunnableLambda(memory.load_memory_variables)
                                | itemgetter(memory.memory_key)),
                "input": RunnablePassthrough() }
                | EntityChain(system_prompt=entity_prompt)
                | RunnableParallel({ 
                    # 라우팅 프롬프트 체인을 구성하여, destination 값을 추출합니다.
                    "destination": ChatPromptTemplate.from_messages([  
                                    # 새로운 라우터 체인: route_chain과 기존의 키들을 합쳐서 dispatcher에 전달합니다.
                                    SystemMessage(content=system_prompt + "\n최종적으로 다음으로 실행해야 하는 Step을 결정하세요."),
                                    MessagesPlaceholder("history"),
                                    ("human", "Screened Intents:\n"
                                              "{intent}\n"
                                              "Utterance: {question}") ])
                                | get_llm().with_structured_output(RouterQuery)
                                | RunnableLambda(lambda x: x.destination),
                    "context": itemgetter("result")
                            | self.get_adaptive_retriever(
                                vectorstore=vectorstore.vectorstore,
                                compressor=vectorstore.reranker)
                            | self._process_context,
                    "question": itemgetter("question"),
                    "intent": itemgetter("intent"),
                    "parsed_intent": itemgetter("parsed_intent"),
                    "history": itemgetter("history") })
                | RunnablePassthrough.assign(
                    context=RunnableLambda(lambda x: x["context"]["context"]),
                    raw_context=RunnableLambda(lambda x: x["context"]["raw_context"]),
                    raw_treatment=RunnableLambda(lambda x: [data.metadata.get("치료") for data in x["context"]["raw_context"]]))
                | StepDispatcher(
                    system_prompt=system_prompt, 
                    timer_prompt=timer_prompt)
            )
