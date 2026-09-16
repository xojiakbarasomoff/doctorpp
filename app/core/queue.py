from functools import lru_cache
from typing import cast

from arq.connections import ArqRedis

from app.core.config import get_settings


@lru_cache
def get_arq_pool() -> ArqRedis:
    """Lazy, lru_cache'd ArqRedis pool — same pattern as get_engine() in
    app.core.db. ArqRedis.from_url() doesn't connect eagerly (unlike
    arq.connections.create_pool(), which pings Redis at creation time), so
    this stays a plain sync singleton rather than needing FastAPI
    lifespan/app.state wiring.
    """
    # redis-py's Redis.from_url (which ArqRedis inherits) has no return type
    # annotation, so mypy sees Any here — cast at that boundary.
    return cast(ArqRedis, ArqRedis.from_url(get_settings().redis_url))
