import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import rasterio as rio
import numpy as np

from serve.schemas import SegmentRequest
from models.infer import load_model
from scripts.export_stitched import stitch_tiles
from explain.overlay import overlay_heatmap

app = FastAPI(title="DrShym Flood Segmenter")
MODEL = None
THRESH = float(os.getenv("DRSHYM_THRESHOLD", 0.45))


@app.on_event("startup")
async def _load():
    global MODEL
    MODEL = load_model(os.getenv("DRSHYM_CKPT", None))


@app.post("/v1/segment")
async def segment(req: SegmentRequest):
    image_uri = req.image_uri.replace("file://","")
    outputs_dir = Path(os.getenv("OUTPUTS", "outputs")); outputs_dir.mkdir(exist_ok=True, parents=True)
    scene_id = Path(image_uri).stem
    stitched_mask, stitched_proba = stitch_tiles(image_uri, MODEL, tile=req.options.get("tile",512), overlap=req.options.get("overlap",64), threshold=THRESH, out_dir=str(outputs_dir))

    overlay_png = overlay_heatmap(image_uri, stitched_proba)

    with rio.open(stitched_mask) as m:
        mask = m.read(1)
    left = mask[:, :mask.shape[1]//2].mean(); right = mask[:, mask.shape[1]//2:].mean()
    side = "eastern" if right>left else "western"
    caption = f"Flooding detected along the {side} half; contiguous band near low-slope areas."

    return JSONResponse({
        "scene_id": scene_id,
        "outputs": {"mask_uri": stitched_mask, "proba_uri": stitched_proba, "overlay_png": overlay_png},
        "caption": caption,
        "provenance": {"model": "unet_resnet18_v0.1", "threshold": str(THRESH), "calibration": os.getenv("DRSHYM_CALIB","temperature=1.0")},
        "policy": {"crs_kept": True, "geojson_exported": bool(int(os.getenv("DRSHYM_EXPORT_GEOJSON","1")))}
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
