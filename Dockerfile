FROM python:3.11-slim

# System dependencies needed by PyMuPDF and pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    mupdf-tools \
    poppler-utils \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p uploads data/chromadb static

# Expose port (Railway injects $PORT at runtime)
EXPOSE 8000

# Start the app
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
