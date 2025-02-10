from ray import serve

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSerializable
from langchain_core.messages import SystemMessage

from langchain_neo4j import GraphCypherQAChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser

from operator import itemgetter
from typing import Dict

from app.core.langchain_module.rag import HybridRAG
from app.core.langchain_module import DDG_LLM
from app.core.prompts.follow_up_care import system_prompt
from app.service.medical_inquiry import MedicalInquiryService


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10)
class FollowupCareService(MedicalInquiryService):
    system_prompt = system_prompt

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_rag_chain(
        self,
        vectorstore: HybridRAG,
        compressor: CrossEncoderReranker,
        neo4j_chain: GraphCypherQAChain,
        memory: ConversationBufferMemory
    ) -> RunnableSerializable:
        
        def _combine_context(step_output: Dict) -> Dict:
            # Vector 검색 결과 처리
            vector_results = []
            for doc in step_output['vector_context']:
                source_data = doc.page_content.strip()
                metadata_text = "\n".join([f"{k}:{v}" for k, v in doc.metadata.items()])
                vector_results.append(f"{source_data}\n{metadata_text}")
            
            # Neo4j 결과 가져오기
            neo4j_result = step_output['neo4j_context'].get("result", "")
            vector_result = "\n".join(vector_results)
            # 컨텍스트 통합
            print(neo4j_result)
            combined_context = (
                "지식 그래프 컨텍스트:\n"
                f"{neo4j_result}\n\n"
                "벡터 검색 컨텍스트:\n"
                f"{vector_result}\n\n")
            return {
                "context": combined_context,
                "question": step_output['question'],
                "history": step_output['history']
            }
        
        vector_retriever = self.get_adaptive_retriever(
            vectorstore=vectorstore,
            compressor=compressor)
        # 통합 Chain 구성
        return (
            RunnableParallel({
                "hospital": itemgetter("hospital"),
                "treatment": itemgetter("treatment"),
                "question": itemgetter("question"),
                "vector_context": itemgetter("question") | vector_retriever,
                "neo4j_context": RunnablePassthrough.assign(
                    query=itemgetter("question"),
                    hospital=itemgetter("hospital"),
                    treatment=itemgetter("treatment"))
                    | neo4j_chain,
                "history": RunnablePassthrough.assign(
                    history=RunnableLambda(memory.load_memory_variables)
                            | itemgetter(memory.memory_key)) 
                    | itemgetter("history")})
            | _combine_context
            | ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder("history"),
                ("human", "컨텍스트: \n"
                          "{context}\n\n"
                          "방문했던 기관: {hospital}\n"
                          "진료질환: {treatment}\n"
                          "문의내용: {question}"),
                ])
            | self.llm
            | StrOutputParser()
        )
