# import argparse
# from pathlib import Path
# import rasterio as rio
# import numpy as np
# import torch

# from models.infer import load_model


# def main(in_dir, ckpt, out_dir):
#     model = load_model(ckpt)
#     out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
#     for tif in Path(in_dir).glob('*.tif'):
#         with rio.open(tif) as src:
#             arr = src.read(1).astype('float32')
#             x = torch.from_numpy(arr)[None,None]
#             with torch.inference_mode():
#                 p = torch.sigmoid(model(x)).squeeze().numpy().astype('float32')
#             profile = src.profile.copy(); profile.update(dtype='float32')
#         o = out / (tif.stem + '.proba.tif')
#         with rio.open(o, 'w', **profile) as dst:
#             dst.write(p,1)

# if __name__ == '__main__':
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--ckpt', required=True)
#     ap.add_argument('--in', dest='inp', required=True)
#     ap.add_argument('--out', required=True)
#     args = ap.parse_args()
#     main(args.inp, args.ckpt, args.out)

# scripts/predict_folder.py
# Usage:
#   python -m scripts.predict_folder --in data/scenes --out outputs
# or
#   python scripts/predict_folder.py --in data/scenes --out outputs
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

# Ensure repo root is on sys.path when called as a script
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import rasterio

from models.unet import UNet
from models.encoder_backbones import make_encoder
from utils.seed import fix_seeds


def make_unet_flexible(encoder_name: str = "resnet18", device: str = "cpu") -> torch.nn.Module:
    """Create UNet with a variety of possible constructor signatures.
    Tries (num_classes=1) then (n_classes=1) then (out_channels=1), then no kwarg."""
    enc, channels = make_encoder(encoder_name)
    model = None
    last_err = None

    for kwargs in ({"num_classes": 1}, {"n_classes": 1}, {"out_channels": 1}, {}):
        try:
            model = UNet(enc, channels, **kwargs)
            break
        except TypeError as e:
            last_err = e

    if model is None:
        raise TypeError(
            f"Could not instantiate UNet with any of the common output-arg names. Last error: {last_err}"
        )

    model.to(device)
    model.eval()
    return model


def load_unet(ckpt_path: str | None, encoder_name: str = "resnet18", device: str = "cpu") -> torch.nn.Module:
    """Load UNet and optionally a checkpoint. If no ckpt, return deterministic random model (for CI demo)."""
    model = make_unet_flexible(encoder_name=encoder_name, device=device)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict mismatches -> missing={missing}, unexpected={unexpected}")
    else:
        fix_seeds(42)
    return model


def read_tif(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile


def normalize(arr: np.ndarray) -> np.ndarray:
    mean = float(arr.mean())
    std = float(arr.std() + 1e-6)
    return (arr - mean) / std


def write_geotiff(path: str, data: np.ndarray, profile: dict, dtype: str):
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, compress="deflate", bigtiff="IF_SAFER")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(dtype), 1)


@torch.inference_mode()
def infer_tile(model: torch.nn.Module,
               tile_fp: pathlib.Path,
               out_dir: pathlib.Path,
               threshold: float,
               device: str = "cpu") -> Tuple[pathlib.Path, pathlib.Path]:
    arr, profile = read_tif(str(tile_fp))
    x = torch.from_numpy(normalize(arr)[None, None, ...]).to(device)  # (1,1,H,W)

    # --- NEW: adapt channels if encoder expects 3 ---
    try:
        expected_c = None
        # try to infer from the first Conv2d we find
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                expected_c = m.in_channels
                break
        if expected_c is None:
            expected_c = 1  # be safe
        if x.shape[1] == 1 and expected_c == 3:
            x = x.repeat(1, 3, 1, 1)  # (1,3,H,W)
        elif x.shape[1] == 3 and expected_c == 1:
            x = x[:, :1, :, :]
    except Exception:
        # conservative fallback for ResNet-style encoders
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
    # --- end NEW ---

    logits = model(x)                            # (1,1,H,W) or compatible
    proba = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (proba >= float(threshold)).astype(np.uint8)

    stem = tile_fp.stem
    proba_fp = out_dir / f"{stem}_proba.tif"
    mask_fp  = out_dir / f"{stem}_mask.tif"

    write_geotiff(str(proba_fp), proba, profile, dtype="float32")
    write_geotiff(str(mask_fp),  mask,  profile, dtype="uint8")
    return proba_fp, mask_fp

if __name__ == "__main__":
    main()
