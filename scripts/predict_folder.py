import argparse
from pathlib import Path
import rasterio as rio
import numpy as np
import torch

from models.infer import load_model


def main(in_dir, ckpt, out_dir):
    model = load_model(ckpt)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for tif in Path(in_dir).glob('*.tif'):
        with rio.open(tif) as src:
            arr = src.read(1).astype('float32')
            x = torch.from_numpy(arr)[None,None]
            with torch.inference_mode():
                p = torch.sigmoid(model(x)).squeeze().numpy().astype('float32')
            profile = src.profile.copy(); profile.update(dtype='float32')
        o = out / (tif.stem + '.proba.tif')
        with rio.open(o, 'w', **profile) as dst:
            dst.write(p,1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    main(args.inp, args.ckpt, args.out)
