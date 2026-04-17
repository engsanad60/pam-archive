FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/chromadb uploads logs static

EXPOSE 8000

CMD ["sh", "-c", "python main.py"]
