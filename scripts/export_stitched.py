import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from utils.geo import blend_weight
import torch


def stitch_tiles(scene_tif: str, model, tile=512, overlap=64, threshold=0.45, out_dir='outputs') -> Tuple[str, str]:
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    with rio.open(scene_tif) as src:
        H, W = src.height, src.width
        profile = src.profile.copy(); profile.update(count=1, dtype='float32')
        prob = np.zeros((H,W), dtype='float32'); wsum = np.zeros((H,W), dtype='float32')
        win = blend_weight(tile, overlap)
        for y in range(0, max(H - tile + 1,1), tile-overlap):
            for x in range(0, max(W - tile + 1,1), tile-overlap):
                window = Window(x,y,tile,tile)
                arr = src.read(1, window=window).astype('float32')
                with torch.inference_mode():
                    p = torch.sigmoid(model(torch.from_numpy(arr)[None,None])).squeeze().numpy().astype('float32')
                y2=min(y+tile,H); x2=min(x+tile,W)
                h=y2-y; w=x2-x
                prob[y:y2, x:x2] += p[:h,:w]*win[:h,:w]
                wsum[y:y2, x:x2] += win[:h,:w]
        prob /= (wsum+1e-6)
    proba_path = str(outp / f"{Path(scene_tif).stem}_proba.tif")
    mask_path = str(outp / f"{Path(scene_tif).stem}_mask.tif")
    with rio.open(proba_path, 'w', **profile) as dst:
        dst.write(prob,1)
    mprofile = profile.copy(); mprofile.update(dtype='uint8')
    with rio.open(mask_path, 'w', **mprofile) as dst:
        dst.write((prob>=threshold).astype('uint8'),1)
    return mask_path, proba_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--proba_dir')
    ap.add_argument('--scene_meta')
    ap.add_argument('--out', default='outputs/scenes')
    args = ap.parse_args()
