import os
from typing import Dict
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import LoggingConfig

from app.api.controller import MedicalInquiryRouterIngress
from app.service.medical_inquiry import MedicalInquiryService

from app.core.lifespan import service_lifecycle


app = FastAPI(
    title="Dencomm Medical Inquiry API",
    summary="Dencomm Medical Inquiry API",
    description="Dencomm Medical Inquiry API",    
    author="Conan",
    version="0.1.0",
    # lifespan=service_lifecycle
    )

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])


@serve.deployment(
    placement_group_bundles=[{
        "CPU": 1.0, 
        }], 
    placement_group_strategy="STRICT_PACK",
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10)
@serve.ingress(app=app)
class APIIngress(
    MedicalInquiryRouterIngress
    ):
    routing = False

    def __init__(
        self, 
        service: DeploymentHandle = None,
    ) -> None:
        super().__init__(service=service)

        self.server_logger.info("""
            ####################
            #  Server Started  #
            ####################
        """)
        self.server_logger.info(app.routes)

        # 추가 라우터가 있으면 계속 등록
        for cls in self.__class__.mro():
            if hasattr(cls, "routing"):
                if cls.routing:
                    cls.service = self.service
                    cls.service_as_stream = self.service_as_stream
                    cls.register_routes(self=cls)
                    app.include_router(
                        cls.router,
                        prefix=cls.prefix,
                        tags=cls.tags,
                        include_in_schema=cls.include_in_schema)
                    self.server_logger.info(f"Routing {cls.prefix} to Application Updated from {cls.__name__}")

    @app.get("/health")
    async def healthcheck(self,):
        try:
            return Response(
                content=json.dumps({"message": "ok"}),
                media_type="application/json",
                status_code=200)
        except Exception as e:
            self.server_logger.error("error" + e)
            return Response(
                    content=f"Server Status Unhealthy",
                    status_code=500
                )


def build_app(
    cli_args: Dict[str, str]
) -> serve.Application:
    serve.start(
        proxy_location="EveryNode", 
        http_options={
            "host": "0.0.0.0",
            "port": cli_args.get("port", 8504),
            # "location": "EveryNode"
            },
        logging_config=LoggingConfig(
            log_level="INFO",
            logs_dir="./logs",
            enable_access_log=True)
        )
    return APIIngress.options(
        ).bind(
            service=MedicalInquiryService.options(
                ).bind(
                    # llm=app.state.global_llm
                )
            )
