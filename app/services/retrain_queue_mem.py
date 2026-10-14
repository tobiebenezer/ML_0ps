"""Fallback in-memory retrain queue when Redis is down."""
from collections import deque

queue: deque[str] = deque()
params: dict[str, dict] = {}


def append(job_id: str) -> None:
    queue.append(job_id)


def append_params(job_id: str, p: dict) -> None:
    params[job_id] = p


def popleft() -> str | None:
    return queue.popleft() if queue else None
