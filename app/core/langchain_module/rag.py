import uuid
from typing import List, Dict, Optional, Any

import torch

from qdrant_client import QdrantClient
from qdrant_client.http import models

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts.prompt import PromptTemplate

from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from app.core.database.neo4j import NEO4J_URL, NEO4J_AUTH
from app.core.langchain_module.llm import DDG_LLM
from app.core.prompts.follow_up_care import system_prompt, neo4j_query_prompt, graph_rag_prompt


class VectorStore:
    def __init__(self, collection_name: str = "test_collection"):
        self.collection_name = collection_name
        self.client = self._init_client()
        self.embeddings = self._get_embeddings()
        self.embedding_dimensions = self._get_embedding_dimensions()
        self.vectorstore = self._init_vectorstore()
        self.reranker = self._get_reranker()
        self._ensure_collection()       

    def _init_vectorstore(self):
        return Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    def _init_client(self):
        return QdrantClient(host="qdrant_vdb", port=6333)  # Qdrant 서버 주소
        # return QdrantClient(":memory:")  # 메모리에서 실행 (테스트용)

    def _get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3", # dragonkue/BGE-m3-ko
            # model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            model_kwargs={'device': 'cpu'}
            )

    def _get_embedding_dimensions(self):
        return self.embeddings._client.get_sentence_embedding_dimension()

    def _get_reranker(self):
        return CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-v2-m3", # sridhariyer/bge-reranker-v2-m3-openvino
                # model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                model_kwargs={'device': 'cpu'}
                ),
            top_n=2
            )

    def _ensure_collection(self):
        """컬렉션이 없으면 생성"""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimensions,  # 임베딩 차원
                    distance=models.Distance.COSINE
                )
            )

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """텍스트를 벡터로 변환하여 저장"""
        if not text.strip():
            return False

        try:
            vector = self.embeddings.embed_query(text)
            _metadata = {
                "_id": uuid.uuid4().hex
            }
            if metadata:
                _metadata.update(metadata)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=vector,
                        payload={
                            # "text": text,
                            "page_content": text,
                            "metadata": _metadata
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"Error adding text: {e}")
            return False

    def search(self, query: str, limit: int = 50) -> List[Dict]:
        """텍스트로 유사한 문서 검색"""
        try:
            search_vector = self.embeddings.embed_query(query)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=search_vector,
                limit=limit,
                score_threshold=0.7
            )
            return [
                {
                    "text": result.payload["page_content"],
                    "metadata": result.payload["metadata"],
                    "score": result.score
                }
                for result in results
            ]
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.model_dump()
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self) -> bool:
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
        
    def delete_item(self, item_id: str) -> bool:
        """아이템 삭제"""
        try:
            r = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[item_id],
                    wait=True # 삭제 작업이 완료될 때까지 대기
                )
            return True
        except Exception as e:
            print(f"Error deleting item: {e}")
            return False

    def as_retriever(self, **kwargs):
        """LangChain 리트리버 반환"""
        return self.vectorstore.as_retriever(**kwargs)


class HybridRAG(VectorStore):
    def __init__(self, collection_name: str = "test_collection2", llm: DDG_LLM = DDG_LLM()):
        super().__init__(collection_name)
        self.neo4j_graph = self._init_neo4j()
        self.neo4j_chain = self._init_neo4j_chain()
        self.llm = llm

    def _init_neo4j(self):
        return Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_AUTH[0],
            password=NEO4J_AUTH[1]
        )

    def _init_neo4j_chain(self):
        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.neo4j_graph,
            verbose=True,
            validate_cypher=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            cypher_prompt=PromptTemplate(
                input_variables=["schema", "question"],
                template=neo4j_query_prompt
            ),
            qa_prompt=PromptTemplate(
                input_variables=["context", "question"],
                template=graph_rag_prompt
                )
            )
        
