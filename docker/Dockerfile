FROM python:3.10-slim

# System deps (minimal); rasterio wheels include GDAL, but gdal-bin is handy for tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin curl && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /app
COPY . /app

# Use official CPU wheels for PyTorch; then install the rest
RUN python -m pip install --upgrade pip && \
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install fastapi uvicorn[standard] pydantic rasterio shapely pyproj numpy scipy matplotlib pyyaml

EXPOSE 8080
CMD ["python", "serve/api.py"]
