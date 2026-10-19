FROM python:3.12-slim

# Install Tesseract (system dependency — the whole point of containerizing)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Pre-install Python dependencies into the system Python
# (avoids uv creating a venv at runtime, which would re-download on each build)
RUN uv pip install --system pymupdf Pillow numpy pytesseract opencv-python-headless

WORKDIR /app
COPY normalize_pdfs.py .
