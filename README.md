# DrShym Climate — Flood Extent Mapping (MVP)

**Objective**: Detect flooded vs non‑flooded land from Sentinel‑1 SAR tiles; produce georeferenced masks with calibrated confidence; explainability overlays; exportable outputs.

## Quickstart (One Command)
```bash
docker compose -f docker/docker-compose.yml up --build
# Service on http://localhost:8080
```

### Example Request
```bash
curl -X POST http://localhost:8080/v1/segment   -H 'Content-Type: application/json'   -d '{
    "domain": "flood_sar",
    "image_uri": "file:///data/scenes/S1_scene_001.tif",
    "options": {"tile":512, "overlap":64, "explain": true}
  }'
```

### Response
Returns URIs for `mask`, `proba`, and `overlay_png`, plus a brief caption and provenance.

## Pipeline
- **ingest**: reads GeoTIFF, tiles with CRS kept, writes DrShymRecord JSON.
- **model**: UNet (ResNet18 encoder). Train with `scripts/train.py --config configs/flood.yaml`.
- **weak supervision**: generate pseudo‑labels with `scripts/predict_folder.py` then curate uncertain tiles.
- **stitch+export**: blends tiles to full‑scene GeoTIFF.
- **serve**: FastAPI `/v1/segment` for deterministic inference.
- **explainability**: produces PNG overlays; governance blocks numeric area claims.

## Metrics & Calibration
- IoU, F1, precision, recall, Brier, ECE.
- Temperature scaling.

## Data Notes
- Sentinel‑1 IW GRD, VV. Public flood polygons or hand‑labels (100+ tiles).

## Reproducibility
- Seed fixed; `PYTHONHASHSEED=0` in Docker.

## Tests
```bash
pytest -q
```

## CLI
```bash
python scripts/train.py --config configs/flood.yaml
python scripts/predict_folder.py --ckpt artifacts/checkpoints/best.pt --in data/tiles/test --out outputs/tiles
python scripts/export_stitched.py --proba_dir outputs/tiles --scene_meta data/scenes/meta.json --out outputs/scenes
```

## GPU Note
Swap Docker base to CUDA variant if you want GPU inference/training.
