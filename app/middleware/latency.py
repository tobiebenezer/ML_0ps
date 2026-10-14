"""Latency middleware: times every /predict call into monitoring_service."""
import time

from starlette.middleware.base import BaseHTTPMiddleware


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api/v1/predict"):
            from app.services import monitoring_service

            monitoring_service.record_latency_ms((time.perf_counter() - start) * 1000)
        return response
