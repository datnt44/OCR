FROM python:3.11-slim

# Cài đặt Tesseract OCR, ZBar, và thư viện đồ họa cập nhật (libgl1 thay thế libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-vie \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
