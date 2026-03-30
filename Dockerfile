FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application file
COPY main.py .

# Diagnostic: confirm files are present in container
RUN echo "=== FILES AFTER COPY ===" && ls -la /app/ && echo "=== END FILES ==="

# Expose port for Cloud Run
EXPOSE 8080

CMD ["python", "main.py"]
