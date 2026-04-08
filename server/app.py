"""
server/app.py — re-exports the FastAPI app for multi-mode deployment.

The openenv validator expects a `server.app` module. This file simply
imports and re-exports the FastAPI `app` object from server.server so
that both `uvicorn server.app:app` and `uvicorn server.server:app` work.
"""
from server.server import app  # noqa: F401

__all__ = ["app"]
