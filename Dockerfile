FROM python:3.11-slim

# Cài đặt Tesseract OCR, gói tiếng Việt, thư viện ZBar, OpenCV system dependencies và PostgreSQL dev libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-vie \
    libzbar0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
