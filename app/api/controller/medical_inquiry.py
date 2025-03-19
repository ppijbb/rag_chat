import json

from typing import Any, List
import time
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, Response, JSONResponse

from ray import serve

from app.service.medical_inquiry import MedicalInquiryService
from app.model.dto.medical_inquiry import ChatRequest, ChatResponse
from app.api.controller.base_router import BaseIngress
from app.api.controller.descriptions.medical_inquiry import chat_description

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
            description=chat_description)
        async def medical_inquiry_chat(
            request: ChatRequest,
        )->JSONResponse:
            result = {
                "text": "", 
                "screening": None, 
                "treatment": None
            }
            state = 0
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
                end = time.time()
                self.server_logger.debug(f"Result in {end - st}sec: {result}")
                # ----------------------------------- #
                assert len(result) > 0, "Generation failed"
                status_code = 200
                content = ChatResponse(language=request.lang, **result)
                content.state = await self.service.get_state.remote(content)
            except AssertionError as e:
                self.server_logger.error(f"!!! data validation error !!! {e}")
                result["text"] = str(e)
                status_code = 501
                content = ChatResponse(**result)
            except Exception as e:
                err_msg =traceback.print_exc()
                self.server_logger.error(f"!!! unkwon error !!! {err_msg}")
                result["text"] = "Generation failed"
                status_code = 500
                content = ChatResponse(**result)
            finally:
                return JSONResponse(
                    status_code=status_code, 
                    content=content.model_dump()
                )

        @router.post(
            "/chat/stream",
            description=chat_description,
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
