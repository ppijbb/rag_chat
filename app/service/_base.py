from abc import ABC, abstractmethod
import shelve
from typing import Any, Dict, List
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSerializable

from app.core.langchain_module.rag import VectorStore

class BaseService(ABC):

    def _get_user_history(
        self, 
        memory_key:str
    ) -> List:
        with shelve.open(f'{self._memory_path}/{memory_key}') as db:
            data = [] if "history" not in db.keys() else db["history"]
        return data
    
    def _add_user_history(
        self, 
        memory_key:str, 
        data: Any
    )->None:
        with shelve.open(f'{self._memory_path}/{memory_key}') as db:
            if isinstance(data, (tuple, list)):
                for d in data:
                    self.logger.warning(d)
                    db["history"] += [d]
            else:
                db["history"] += [data]

    @abstractmethod
    def get_rag_chain(
        self,
        vectorstore: VectorStore, 
        memory: ConversationBufferMemory
    )-> RunnableSerializable:
        pass

