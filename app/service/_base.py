from abc import ABC, abstractmethod
import shelve
from typing import Any, Dict, List
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSerializable

from app.core.langchain_module.rag import VectorStore
from app.core.logging import get_logger

class BaseService(ABC):
    service_logger = get_logger()
    
    def _get_user_history(
        self, 
        memory_key:str,
        state: int
    ) -> List:
        with shelve.open(f'{self._memory_path}/{memory_key}') as db:
            db["history"] = [] if "history" not in db.keys() or state == 0 else db["history"]
            data = db["history"]
        return data
    
    async def _add_user_history(
        self, 
        memory_key:str,
        data: Any,
        state: int
    )->None:
        with shelve.open(f'{self._memory_path}/{memory_key}') as db:
            if state == 0:
                db["history"] = []
            if isinstance(data, (tuple, list)):
                for d in data:
                    db["history"] += [d] # append 사용하면 느리고 처리도 잘 안되는 경우 발생
            else:
                db["history"] += [data]

    @abstractmethod
    def get_rag_chain(
        self,
        vectorstore: VectorStore, 
        memory: ConversationBufferMemory
    )-> RunnableSerializable:
        pass

