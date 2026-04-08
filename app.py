"""
HuggingFace Spaces / Docker entry point.
Starts the DataGuard FastAPI server on port 7860.
"""
import os
import sys

# Make sure the repo root is on the path so `server.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
        log_level="info",
    )