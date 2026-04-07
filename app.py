"""
HuggingFace Spaces entry point.
HF Spaces expects either app.py or a Dockerfile.
This file lets the Space run without Docker by booting the FastAPI server directly.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

import uvicorn
from server import app  # noqa: F401  (imported for HF Spaces direct-run mode)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )