"""Optional FastAPI receiver for the event dispatcher (``server`` extra).

Kept dependency-light: FastAPI is imported lazily inside the builder, so the
core package installs without it. Point your GitHub/Jira/deploy webhooks at
``POST /webhooks/{source}``.
"""
from __future__ import annotations

from typing import Any

from .dispatcher import EventDispatcher, SignatureError, UnknownSourceError


def build_fastapi_app(dispatcher: EventDispatcher, *, secret: str | None = None) -> Any:
    """Build a FastAPI app exposing ``POST /webhooks/{source}``."""
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI(title="Coalent Event Receiver")

    @app.post("/webhooks/{source}")
    async def receive(source: str, request: Request) -> dict[str, int]:
        raw = await request.body()
        payload = await request.json()
        signature = request.headers.get("X-Hub-Signature-256")
        try:
            events = dispatcher.dispatch(
                source, payload, raw_body=raw, signature=signature, secret=secret
            )
        except UnknownSourceError as exc:
            raise HTTPException(status_code=404, detail=f"unknown source: {source}") from exc
        except SignatureError as exc:
            raise HTTPException(status_code=401, detail="invalid signature") from exc
        return {"accepted": len(events)}

    return app
