from abc import ABC
from operator import itemgetter
from typing import List
import datetime

from ray import serve

from langchain.schema import Document, StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MultiQueryRetriever

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSerializable, Runnable

from langchain_community.retrievers import BM25Retriever

from app.core.langchain_module.rag import VectorStore
from app.core.langchain_module.llm import DDG_LLM
from app.util.time_func import format_datetime_with_ampm


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
    dental_section_list = [
        "혀", "입천장", 
        "좌측 턱", "우측 턱", 
        "상악 좌측치", "하악 좌측치", 
        "상악 전치부", "하악 전치부", 
        "상악 우측치", "하악 우측치"]
    system_prompt: str =f"""
# Service Informations
현재 시각: {format_datetime_with_ampm(datetime.datetime.now())}

# Task
사용자의 Utterance에서 항목들을 정리해야합니다.
1차적으로 Screening 된 정보와 이전까지 기록된 항목들을 바탕으로 질문과 증상을 정리하세요.
최종적으로 정리된 표와 주어지는 Context를 보고 질문에 대해 적합한 치료 방법과 치료 시간을 제시하세요.

## Screening 
- 증상 : (발화 중 드러난 증상 작성. 의학적 용어로 작성.)
- 지속 기간 : (발화한 기간을 작성. 시간의 형태로 작성.) 
- 증상 부위 : (증상의 위치가 구체적으로 특정된 경우에 {', '.join(dental_section_list)} 중에서 다중 선택)
- 증상 강도 : (0:통증/불편 없음 1-2:가벼운 통증/불편 3-4:보통 수준의 통증/불편 5-6:심한 통증/불편 7-8:매우 심한 통증/불편 9-10:극심한 통증/불편)
- 증상 유발요인 : (발화한 유발 요인, 상황 작성. 보편적으로 증상이 발생할 수 있는 경우를 유추하여 작성.)
- 하고 싶은 말 : (따로 없는 경우는 작성하지 않음) 

# Tone And Manner
- 친절하고 이해하기 쉽게 질문하고 답변해야합니다.
- 항목별로 구체적인 항목을 말할 수 있게 질문해야합니다.
- 친근한 어조의 발화 형식으로 질문하고 답변해야합니다.
- 아주 짧은 문장으로 응답해야만 합니다.

# Workflow
Task를 수행하기 위한 3 단계에 대한 가이드라인입니다.

## Step1. Screening
- Screening 정보는 추가적으로 작성되어 사용자에게 지속적으로 전달합니다.
- Screening 정보는 시스템적으로 처리 가능한 수치, 문자의 형태로 고정합니다.
- 이전 채팅에서 Screening 된 history는 반드시 유지되어야 합니다.
- Screening에 작성 되지 않는 항목이 있는 경우에만 추가적으로 해당하는 항목에 대한 질문을 수행합니다.(하고싶은 말 제외)
    - 질문을 할 때는 증상, 지속 기간, 증상 부위, 증상 강도, 증상 유발요인 중 비어있는 항목 1개에 대해서만 1회 생성합니다.
    - 질문을 할 때는, 항목에 대한 명시를 하지 않고 질문만 생성합니다.
    - 질문 항목에 대해 사용자가 뚜렷하게 모를 수 있으므로, 모른다는 응답에도 해당 항목은 작성되어야합니다.
- 만약 Screening 표가 모두 작성된 경우(하고 싶은 말 제외), Step1는 즉시 종료합니다.

## Step2. Chat with Context
- Step1이후 Context에 기반한 예상 치료 정보를 제공합니다.
- Screening에서 모든 항목이 채워진 경우에 예상 치료 정보를 제공합니다.
- Context가 없는 경우, 치료 방법과 치료 시간은 제시하지 말아야합니다.
- Context의 예시를 참고하여 예상되는 치료 방법과 시간을 제시합니다.
- 제공받은 Case들에 대해서 복합 치료가 필요한 경우 두 가지 방법 모두 제시하세요.
- 1회 공지 이후 Step2는 종료됩니다.

## Step3. Reservations and Scheduling
- Step2 이후에는 예약 및 스케줄링에 대한 정보를 제공합니다.
- 사용자에게 방문 시간에 대해 질문하고 해당 시간과 날짜를 판단하여 제공합니다.
- 현재 시각 기준으로 예약 날짜를 판단하며 정확한 날짜의 형식으로 제공합니다. (ex. 2024년 11월 21일 목요일 오전 9시 30분 등...)

# Output
- (필수)Screening 표
- (Step1 진행 중)Screening 항목에 대한 질문 질문
- (Step2 진행 중)예상 치료 방법
- (Step3 진행 중)사용자가 원하는 방문 시간
""".strip()

    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.llm = kwargs.get("llm", DDG_LLM())
        self.vectorstore = VectorStore()

    async def inquiry_chat(
        self,
        text:str,
        memory_key: str = "history"
    ):
        rag_chain = self.get_rag_chain(
            vectorstore=self.vectorstore,
            system_prompt=self.system_prompt,
            memory=ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(),
                return_messages=True,
                memory_key=memory_key)
            )
        return await rag_chain.ainvoke({"input": {"question": text}})

    async def inquiry_stream(
        self,
        text:str,
        memory_key: str = "history"
    ):
        rag_chain = self.get_rag_chain(
            vectorstore=self.vectorstore,
            system_prompt=self.system_prompt,
            memory=ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(),
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
                template=(
                    "You are an AI language model assistant. Your task is "
                    "to generate 3 different versions of the given user "
                    "question to retrieve relevant documents from a vector database. "
                    "By generating multiple perspectives on the user question, "
                    "your goal is to help the user overcome some of the limitations "
                    "of distance-based similarity search. Provide these alternative" 
                    "questions separated by newlines.\n\n"
                    "Original question: {question}")
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
    
    # Initialize RAG chain
    def get_rag_chain(
        self,
        vectorstore: VectorStore, 
        system_prompt: str, 
        memory: ConversationBufferMemory
        )-> RunnableSerializable:

        def _process_context(step_output):
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
            print(result)
            return "\n\n".join(result)
        
        retriever = self.get_adaptive_retriever(
            vectorstore=vectorstore.vectorstore,
            compressor=vectorstore.reranker)
        
        # Start of Selection
        chain = (
            {
                "chat_history": RunnablePassthrough.assign(
                    history=RunnableLambda(memory.load_memory_variables) 
                            | itemgetter(memory.memory_key)
                        ),
                "input": RunnablePassthrough()
            }
            | IntentChain()
            | RunnableParallel({
                "context": itemgetter("result") | retriever | _process_context,
                "question": itemgetter("question"),
                "intent": itemgetter("intent"),
                "history": itemgetter("history")
                #   RunnablePassthrough.assign(
                #     history=RunnableLambda(memory.load_memory_variables) 
                #             | itemgetter(memory.memory_key)
                #         )
            })
            | ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder("history"),
                    ("human", "\n"
                              "Contexts:\n{context}\n\n"
                              "Screened Intents:\n{intent}\n\n"
                              "Utterance: {question}"), # HummaMessage 로 넘기는 경우, formatting이 안되는 문제
                ])
            | self.llm
            | StrOutputParser()
        )
        return chain


class IntentChain(Runnable):
    llm = DDG_LLM()
    prompt = ChatPromptTemplate.from_messages([
            # Start of Selection
            ("system", "# Task\n"
                       "사용자의 Utterance에서 [증상, 지속 기간, 증상 부위, 증상 강도, 증상 유발요인, 하고 싶은 말]을 정리해내야 합니다.\n\n"

                       "# Note\n"
                       "- 작성되어야 하는 모든 항목들은 전문적인 단어와 용어로 통일되어야 합니다.\n"
                       "- 발화에서 판단할 수 없는 항목은 작성하지 않고 빈 칸으로 유지합니다.\n"
                       "- 증상은 임상에서 구분 가능한 뚜렷한 명칭으로 표기합니다.\n"
                       "- 뚜렷한 증상 부위의 방향을 명시하거나 증상 부위를 모르는 경우가 아니면 작성되어서는 안됩니다.\n"
                       "- 사용자가 증상 부위를 특정한 경우에만 증상 부위를 작성할 수 있습니다.\n"
                       "- 모른다, 모르겠다 등의 답변도 처리되어야 합니다.\n"
                       "- 증상 강도의 범위는 다음과 같습니다 0:통증/불편 없음, 1-2:가벼운 통증/불편, 3-4:보통 수준의 통증/불편, 5-6:심한 통증/불편, 7-8:매우 심한 통증/불편, 9-10:극심한 통증/불편\n"
                       "- 하고싶은 말은 추가적인 부분으로 굳이 작성될 필요는 없으며 추가적인 특징이 있으면 추가해주세요.\n"
                       "- 발화에서 아래 항목을 모두 작성하여 표로 정리해주세요.\n\n"

                       "# Missing Intent\n"
                       "- 이전에 입력되던 정보는 모두 유지되어야 합니다.\n"
                       "- Screening 표에 채워지지 않은 항목을 리스트로 만듭니다.(하고 싶은 말 제외)\n\n"

                       "# Output Template\n"
                       "|항목|내용|\n"
                       "|---|---|\n"
                       "|증상|(발화 중 드러난 증상 작성 <ex>치아 파절, 구취, 과잉치, 기저질환 등...</ex>)|\n"
                       "|지속 기간|(발화한 기간을 작성 <ex>1개월, 7일 이상 등...</ex> )|\n"
                       "|증상 부위|(증상의 위치가 구체적으로 특정된 경우에 혀, 입천장, 좌측 턱, 우측 턱, 상악 좌측치, 하악 좌측치, 상악 전치부, 하악 전치부, 상악 우측치, 하악 우측치 중에서 다중 선택)|\n"
                       "|증상 강도|(1~10 중에 작성)|\n"
                       "|증상 유발요인|(발화한 유발 요인, 상황 작성<ex>음식물 섭취시, 찬바람 등...</ex>)|\n"
                       "|하고 싶은 말|(따로 없는 경우는 작성하지 않음)|"
                       ),
            MessagesPlaceholder("history"),
            ("user", "Utterance: {question}")
        ])

    def invoke(self, input, config, **kwargs):
        # 입력 데이터를 처리하는 로직 구현
        question = input.get("input", "").get("question", "")
        history = input.get("chat_history", {}).get("history", [])
        # print(input)
        
        chain = self.prompt | self.llm
        intent = chain.invoke({
            "history": history,
            "question": question
        })
        return {
            "result": f"{' '.join([his.content for his in history if his.type=='human'])} {question}", 
            "intent": intent, 
            "question": question, 
            "history": history
        }
    # 필요한 경우 batch, stream 등의 메서드도 구현 가능
    def batch(self, inputs):
        return [self.invoke(input) for input in inputs]
