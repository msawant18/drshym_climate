FROM pytorch/pytorch:2.3.1-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev python3-gdal python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic rasterio shapely pyproj numpy scipy \
    torch torchvision matplotlib pyyaml

EXPOSE 8080
CMD ["python", "serve/api.py"]
