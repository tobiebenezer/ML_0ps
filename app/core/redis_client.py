"""Redis client with graceful fallback (class laptops may not run Redis)."""
import logging
from functools import lru_cache

from app.core.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis():
    try:
        import redis

        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        client.ping()
        log.info("Redis connected at %s:%s", settings.redis_host, settings.redis_port)
        return client
    except Exception as exc:  # noqa: BLE001 - fallback is intentional
        log.warning("Redis unavailable (%s); using in-memory fallback", exc)
        return None


def redis_ping() -> bool:
    client = get_redis()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False
