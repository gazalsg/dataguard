FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY server/ ./server/

# HF Spaces expects the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

USER appuser

# Start FastAPI server
CMD ["python", "server/server.py"]