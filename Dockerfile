FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ONNX embedding model so startup is instant on Railway
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()"

COPY . .
RUN mkdir -p data/chromadb data/texts uploads logs static/images

ENV ANONYMIZED_TELEMETRY=False
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8000
CMD ["python", "main.py"]
