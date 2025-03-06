import json
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from app.service.safe_guard import GaurdService
from app.core.logging import get_logger


class ChatGaurdMiddleware(BaseHTTPMiddleware, GaurdService):
    """
    Custom FastAPI middleware that logs incoming requests,
    measures processing time, and appends a custom header to responses.
    """
    logger = get_logger()

    async def dispatch(
            self, 
            request: Request, 
            call_next: RequestResponseEndpoint
        ) -> Response:
        start_time = time.time()
        self.logger.info(f"Incoming request: {request.method} {request.url.path}")
        
        # Read and parse the request body
        body = await request.body()
        try:
            if request.method == "GET":
                new_body = body
            else:
                data = json.loads(body)
                text = data.get("text", "")
                
                # Use GaurdService to predict
                prediction = self.model.predict([text])[0]
                data["prediction"] = prediction
                
                # Create a new request with the modified body
                new_body = json.dumps(data).encode("utf-8")
            request = Request(request.scope, receive=lambda: new_body)
            
            response = await call_next(request)
        except Exception as exc:
            self.logger.exception("Error while processing request")
            return Response("Internal Server Error", status_code=500)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        response_body = b''
        async for chunk in response.body_iterator:
            response_body += chunk

        # 원본 응답의 상태를 유지하기 위해 body_iterator를 재설정
        async def new_body_iterator():
            yield response_body
        response.body_iterator = new_body_iterator()

        self.logger.info(f"[api log] Response status code: {response.status_code}")
        self.logger.info(f"[api log] Completed {request.method} {request.url.path} from {request.client.host} in {process_time:.3f}s")
        self.logger.info(f"[api log] Request headers: {request.headers}")
        self.logger.info(f"[api log] Request body: {new_body.decode()}")
        self.logger.info(f"[api log] Response headers: {response.headers}")
        self.logger.info(f"[api log] Response body: {response_body.decode()}")
        self.logger.info(f"[api log] ----------------------------------------")

        return response
