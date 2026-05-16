FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-compile --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-compile .

COPY src ./src
COPY models/price_model/v1/car_price_pipeline.pkl ./models/price_model/v1/car_price_pipeline.pkl
COPY models/VINsion/v1/best.pt ./models/VINsion/v1/best.pt

ENV YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
