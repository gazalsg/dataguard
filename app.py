"""
HuggingFace Spaces entry point.
Runs the FastAPI server directly when Docker is not used.
"""
import os
import sys

# Ensure the server package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

import uvicorn
from server import app  # imported so HF Spaces can find `app`

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
        log_level="info",
    )
