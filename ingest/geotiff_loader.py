import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling

from utils.seed import fix_seeds


def zscore(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    mean = float(x.mean())
    std = float(x.std() + 1e-6)
    return (x - mean) / std, {"mean": mean, "std": std}


def minmax01(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    mn = float(np.percentile(x, 1))
    mx = float(np.percentile(x, 99))
    x = np.clip((x - mn) / (mx - mn + 1e-6), 0, 1)
    return x, {"min": mn, "max": mx}


def read_geotiff(path: str, normalize: str = "zscore") -> Dict[str, Any]:
    fix_seeds(42)
    with rio.open(path) as src:
        arr = src.read(1, out_dtype="float32", resampling=Resampling.bilinear)
        profile = src.profile.copy()
        crs = src.crs.to_string() if src.crs else None
        transform = src.transform
    if normalize == "zscore":
        norm, stats = zscore(arr)
    else:
        norm, stats = minmax01(arr)
    return {
        "array": norm.astype("float32"),
        "profile": profile,
        "crs": crs,
        "transform": transform,
        "stats": stats,
    }


def write_drshym_record(tile_path: Path, meta: Dict[str, Any], proc: list, out_json: Path):
    img_id = tile_path.stem
    record = {
        "image_id": img_id,
        "modality": "sentinel1_sar_vv",
        "crs": meta.get("crs"),
        "pixel_spacing": [meta["profile"].get("transform")[0], meta["profile"].get("transform")[4]],
        "tile_size": [meta["profile"].get("width"), meta["profile"].get("height")],
        "bounds": list(rio.transform.array_bounds(meta["profile"]["height"], meta["profile"]["width"], meta["transform"])),
        "provenance": {
            "source_uri": str(meta["profile"].get("source", "")),
            "processing": proc,
        },
        "label_set": ["flooded", "non_flooded"],
    }
    out_json.write_text(json.dumps(record, indent=2))
