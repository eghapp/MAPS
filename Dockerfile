# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY board_classifier_excel_corrected.pkl .
COPY tray_classifier.pkl .

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 60 main:app

# Cache bust: 20260327-1545
