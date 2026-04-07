FROM python:3.11-slim
# Install Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY NWL23.txt .
EXPOSE 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 120 main:app
