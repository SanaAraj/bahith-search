"""Optional query-path tracing via Langfuse.

Tracing is entirely opt-in: it is a no-op unless ``TRACING_ENABLED`` is truthy
*and* the Langfuse SDK and credentials are available. Any failure inside the
tracing code is swallowed and logged — observability must never break a search.

Usage::

    with trace_query("سؤال", mode="hybrid", top_k=5) as tr:
        with tr.span("retrieval"):
            hits = search(...)
        tr.update(output={"num_results": len(hits)})
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    TRACING_ENABLED,
)

logger = logging.getLogger(__name__)

_client: Any | None = None
_init_attempted = False


def _get_client() -> Any | None:
    """Lazily create the Langfuse client, or return None if unavailable."""
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True

    if not (TRACING_ENABLED and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logger.info("Langfuse tracing enabled (host=%s)", LANGFUSE_HOST)
    except Exception:
        logger.warning("Tracing enabled but Langfuse init failed; disabling", exc_info=True)
        _client = None
    return _client


def is_enabled() -> bool:
    return _get_client() is not None


class _NullTrace:
    """Stand-in used when tracing is disabled; every method is a no-op."""

    @contextmanager
    def span(self, name: str, **_: Any) -> Iterator[None]:
        yield None

    def update(self, **_: Any) -> None:
        pass


class _LangfuseTrace:
    def __init__(self, trace: Any) -> None:
        self._trace = trace

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        span = None
        try:
            span = self._trace.span(name=name, metadata=metadata or None)
        except Exception:
            logger.debug("Failed to open span %s", name, exc_info=True)
        try:
            yield None
        finally:
            if span is not None:
                try:
                    span.end()
                except Exception:
                    logger.debug("Failed to end span %s", name, exc_info=True)

    def update(self, **kwargs: Any) -> None:
        try:
            self._trace.update(**kwargs)
        except Exception:
            logger.debug("Failed to update trace", exc_info=True)


@contextmanager
def trace_query(query: str, *, mode: str, top_k: int) -> Iterator[Any]:
    """Open a trace for one query. Yields a trace handle exposing ``span``/``update``."""
    client = _get_client()
    if client is None:
        yield _NullTrace()
        return

    trace = None
    try:
        trace = client.trace(name="search", input={"query": query, "mode": mode, "top_k": top_k})
    except Exception:
        logger.debug("Failed to open trace", exc_info=True)

    if trace is None:
        yield _NullTrace()
        return

    try:
        yield _LangfuseTrace(trace)
    finally:
        try:
            client.flush()
        except Exception:
            logger.debug("Failed to flush Langfuse client", exc_info=True)
