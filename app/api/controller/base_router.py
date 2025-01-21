from abc import ABC, abstractmethod
from typing import List
from logging import Logger
from fastapi import APIRouter
from ray.serve.handle import DeploymentHandle

from app.core.logging import get_logger


class BaseIngress(ABC):
    routing: bool
    prefix: str 
    tags: List[str]
    include_in_schema: bool
    server_logger: Logger = get_logger()
    
    def __init__(
        self, 
        service: DeploymentHandle = None
    ) -> None:
        self.service = service.options() if service else None
        self.service_as_stream = service.options(stream=True) if service else None
    
    @classmethod
    def _get_class(cls):
        return cls
    
    @abstractmethod
    def register_routes(self, router:APIRouter):
        if self.routing:
            self.router = router
        pass

