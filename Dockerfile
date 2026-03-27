FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy minimal main.py (no .pkl files needed)
COPY main_runtime_upload.py main.py

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8080/health'); exit(0 if r.status_code == 200 else 1)" || exit 1

CMD ["python", "main.py"]
