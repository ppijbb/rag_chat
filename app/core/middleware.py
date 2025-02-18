import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from app.service.safe_guard import GaurdService

logger = logging.getLogger("app.core.middleware")

class CustomMiddleware(BaseHTTPMiddleware, GaurdService):
    """
    Custom FastAPI middleware that logs incoming requests,
    measures processing time, and appends a custom header to responses.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("Error while processing request")
            return Response("Internal Server Error", status_code=500)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        logger.info(f"Completed {request.method} {request.url.path} in {process_time:.3f}s")
        return response
