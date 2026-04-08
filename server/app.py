"""
server/app.py - re-exports the FastAPI app and provides main() for multi-mode deployment.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.server import app  # noqa: F401

__all__ = ["app"]


def main():
    import uvicorn
    uvicorn.run(
        "server.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
