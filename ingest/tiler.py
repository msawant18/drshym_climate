from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from ingest.geotiff_loader import write_drshym_record


def sliding_windows(H: int, W: int, tile: int, overlap: int) -> Iterator[Tuple[int, int, int, int]]:
    stride = tile - overlap
    for y in range(0, max(H - tile + 1, 1), stride):
        for x in range(0, max(W - tile + 1, 1), stride):
            yield y, x, tile, tile


def tile_scene(src_path: str, out_dir: str, tile: int = 512, overlap: int = 64, normalize: str = "zscore"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with rio.open(src_path) as src:
        H, W = src.height, src.width
        for y, x, h, w in sliding_windows(H, W, tile, overlap):
            window = Window(x, y, w, h)
            arr = src.read(1, window=window, out_dtype="float32")
            profile = src.profile.copy()
            profile.update({"height": h, "width": w, "transform": rio.windows.transform(window, src.transform)})

            # normalize
            if normalize == "zscore":
                m = arr.mean(); s = arr.std()+1e-6
                arr = (arr - m)/s
            else:
                mn, mx = np.percentile(arr,1), np.percentile(arr,99)
                arr = np.clip((arr-mn)/(mx-mn+1e-6),0,1)
            tpath = out / f"{Path(src_path).stem}_y{y}_x{x}.tif"
            with rio.open(tpath, "w", **profile) as dst:
                dst.write(arr.astype("float32"), 1)
            meta = {"crs": src.crs.to_string() if src.crs else None, "profile": profile, "transform": profile["transform"]}
            write_drshym_record(tpath, meta, ["normalize:"+normalize], tpath.with_suffix(".json"))
