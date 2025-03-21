from typing import List, Any, Dict, AsyncIterator
import datetime
import time

from ray import serve

from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MultiQueryRetriever

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableSerializable

from langchain_community.retrievers import BM25Retriever

from app.core.langchain_module.rag import VectorStore
from app.core.langchain_module.llm import DDG_LLM, get_llm
from app.core.langchain_module.chains.medical_inquiry import ServiceChain
from app.core.prompts.medical_inquiry import MULTI_QUERY_PROMPT
from app.model.dto.medical_inquiry import ChatResponse
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
        
        # Initialize the ServiceChain
        self.service_chain = ServiceChain(
            retriever=self.rag,
            llm=self.llm,
            dental_section_list=self.dental_section_list,
            chain_logger=self.service_logger
        )
        
    async def inquiry_chat(
        self,
        text:str,
        language:str,
        state:int,
        memory_key:str="history"
    ) -> Dict[str, str]:
        start = time.time()
        # history = self._get_user_history(
        #     memory_key=memory_key,
        #     state=state)
        # self.service_logger.info(f"user `{memory_key}` history: {history}")
        
        # Use the ServiceChain instead of get_service_chain
        rag_chain = await self.get_service_chain(
            memory=ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(
                    messages=self._get_user_history(
                        memory_key=memory_key,
                        state=state)),
                return_messages=True,
                memory_key=memory_key),
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

    async def get_state(
        self,
        response: ChatResponse
    ) -> int:
        state = self.vectorstore.search(
            query=response.text,
            collection_name="state_control",
            limit=1)
        return state.pop().get("metadata").get("state") if state else 2 if response.treatment is None else 3

    async def inquiry_stream(
        self,
        text:str,
        language:str,
        state:int,
        memory_key:str="history"
    ) -> AsyncIterator:
        # Use the ServiceChain instead of get_service_chain
        rag_chain = await self.get_service_chain(
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
            include_original=False,
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

    async def get_service_chain(self, memory, language) -> RunnableSerializable:
        return self.service_chain.build_chain(memory=memory, language=language)
