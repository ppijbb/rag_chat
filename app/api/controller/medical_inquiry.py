import json

from typing import Any, List
import time
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, Response

from ray import serve

from app.service.medical_inquiry import MedicalInquiryService
from app.model.dto.medical_inquiry import ChatRequest, ChatResponse
from app.api.controller.base_router import BaseIngress

router = APIRouter()


# @serve.deployment
# @serve.ingress(app=router)
class MedicalInquiryRouterIngress(BaseIngress):
    routing = True
    prefix = "/medical_inquiry"
    tags = ["Medical Inquiry"]
    include_in_schema = True
    
    def __init__(
        self, 
        service: MedicalInquiryService = None
    ) -> None:
        super().__init__(service=service)
        
    @serve.batch(
        max_batch_size=4, 
        batch_wait_timeout_s=0.1)
    async def batched_process(
       self,
       request_prompt: List[Any],
       request_text: List[Any]
    ) -> List[str]:
        self_class = self[0]._get_class() # ray batch wrapper 에서 self가 list로 들어옴
        self_class.server_logger.info(f"Batched request: {len(request_text)}")
        return await self_class.service.summarize.remote(
            input_prompt=request_prompt,
            input_text=request_text,
            batch=True)
    
    def register_routes(self, router:APIRouter=router):
        self.router = router
        
        @router.get("/health")
        async def healthcheck():
            try:
                return Response(
                    content=json.dumps({"message": "ok"}),
                    media_type="application/json",
                    status_code=200)
            except Exception as e:
                self.server_logger.error("error" + e)
                return Response(
                        content=f"Summary Service Can not Reply",
                        status_code=500
                    )

        @router.post(
            "/chat", 
            description='''
            **state 값 정의**\n
            - 0: 시작
            - 1: 증상 부위 입력
            - 2: 진행 중
            - 3: 요약
            - 4: 치료
            - 5: 종료
            ''',
            response_model=ChatResponse)
        async def medical_inquiry_chat(
            request: ChatRequest,
        ) -> ChatResponse:
            result = {"text": "", "screening": None, "treatment": None}
            # Generate predicted tokens
            try:
                # ----------------------------------- #
                st = time.time()
                # result += ray.get(service.summarize.remote(ray.put(request.text)))
                # assert len(request.text ) > 200, "Text is too short"
                chain_result = await self.service.inquiry_chat.remote(
                    # self=self._get_class(),
                    text=request.text,
                    language=request.lang,
                    state=request.state,
                    memory_key=request.uid)
                result.update(chain_result)
                # self.server_logger.info(f"\nRequest: {request.text}")
                self.server_logger.info(f"Result: {result}")
                end = time.time()
                # ----------------------------------- #
                assert len(result) > 0, "Generation failed"
                print(f"Time: {end - st}")
            except AssertionError as e:
                self.server_logger.error("validation error" + e)
                result["text"] = e
            except Exception as e:
                self.server_logger.error("unkwon error" + e)
                result["text"] = "Generation failed"
            finally:
                return ChatResponse(**result)

        @router.post(
            "/chat/stream",
            description='''
            **state 값 정의**\n
            - 0: 시작
            - 1: 증상 부위 입력
            - 2: 진행 중
            - 3: 요약
            - 4: 치료
            - 5: 종료
            ''',
        )
        async def medical_inquiry_chat_stream(
            request: ChatRequest,
        ):
            result = ""
            # Generate predicted tokens
            try:
                # ----------------------------------- #
                st = time.time()
                # result += ray.get(service.summarize.remote(ray.put(request.text)))
                # assert len(request.text ) > 200, "Text is too short"
                return StreamingResponse(
                    content=self.service_as_stream.inquiry_stream.remote(
                        # self=self._get_class(),
                        text=request.text,
                        language=request.lang,
                        state=request.state,
                        memory_key=request.uid),
                    media_type="text/event-stream")
                end = time.time()
                # ----------------------------------- #
                print(f"Time: {end - st}")
            except AssertionError as e:
                self.server_logger.error("validation error" + e)
                result += e
            except Exception as e:
                self.server_logger.error("unkwon error" + e)
                result += "Generation failed"
