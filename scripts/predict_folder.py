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
# scripts/predict_folder.py
# Usage:
#   python -m scripts.predict_folder --in data/scenes --out outputs
#   # or
#   python scripts/predict_folder.py --in data/scenes --out outputs
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

# --- Ensure repo root is importable (so `import models` works when run as a script) ---
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import rasterio

from models.unet import UNet
from models.encoder_backbones import make_encoder
from utils.seed import fix_seeds


# ----------------------------- Model helpers ----------------------------- #
def _instantiate_unet(enc, channels, device: str) -> torch.nn.Module:
    """
    Instantiate UNet trying common constructor kwarg names for binary head.
    Tries: num_classes=1 -> n_classes=1 -> out_channels=1 -> no kwargs.
    """
    last_err = None
    for kwargs in ({"num_classes": 1}, {"n_classes": 1}, {"out_channels": 1}, {}):
        try:
            m = UNet(enc, channels, **kwargs)
            m.to(device)
            m.eval()
            return m
        except TypeError as e:
            last_err = e
    raise TypeError(
        f"Could not instantiate UNet with any common output kwarg names. Last error: {last_err}"
    )


def load_unet(ckpt_path: str | None,
              encoder_name: str = "resnet18",
              device: str = "cpu") -> torch.nn.Module:
    """
    Create UNet(encoder) and optionally load a checkpoint.
    If ckpt_path is None/empty, returns a deterministic randomly-initialized model (for CI/demo).
    """
    enc, channels = make_encoder(encoder_name)
    model = _instantiate_unet(enc, channels, device=device)

    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict mismatches -> missing={missing}, unexpected={unexpected}")
    else:
        fix_seeds(42)  # deterministic random weights

    return model


# ----------------------------- I/O helpers ----------------------------- #
def read_tif(path: str) -> Tuple[np.ndarray, dict]:
    """Read a single-band GeoTIFF -> (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile


def normalize(arr: np.ndarray) -> np.ndarray:
    """Simple z-score normalization with epsilon for safety."""
    mean = float(arr.mean())
    std = float(arr.std() + 1e-6)
    return (arr - mean) / std


def write_geotiff(path: str, data: np.ndarray, profile: dict, dtype: str):
    """Write a single-band GeoTIFF with the same georeference as input."""
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, compress="deflate", bigtiff="IF_SAFER")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(dtype), 1)


# ----------------------------- Inference ----------------------------- #
@torch.inference_mode()
def infer_tile(model: torch.nn.Module,
               tile_fp: pathlib.Path,
               out_dir: pathlib.Path,
               threshold: float,
               device: str = "cpu") -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Run model on one tile and write *_proba.tif and *_mask.tif to out_dir.
    Handles 1-channel SAR by replicating to 3-ch if the encoder requires it.
    """
    # Read & normalize
    arr, profile = read_tif(str(tile_fp))
    x = torch.from_numpy(normalize(arr)[None, None, ...]).to(device)  # (1,1,H,W)

    # --- Channel adaptation: if encoder expects 3 channels, repeat SAR band ---
    try:
        expected_c = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                expected_c = m.in_channels
                break
        if expected_c is None:
            expected_c = 1
        if x.shape[1] == 1 and expected_c == 3:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3 and expected_c == 1:
            x = x[:, :1, :, :]
    except Exception:
        # Fallback for ResNet-like encoders
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
    # -------------------------------------------------------------------------

    # Forward
    logits = model(x)                          # (1,1,H,W) or compatible
    proba = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (proba >= float(threshold)).astype(np.uint8)

    # Write outputs
    stem = tile_fp.stem
    proba_fp = out_dir / f"{stem}_proba.tif"
    mask_fp  = out_dir / f"{stem}_mask.tif"

    write_geotiff(str(proba_fp), proba, profile, dtype="float32")
    write_geotiff(str(mask_fp),  mask,  profile, dtype="uint8")
    return proba_fp, mask_fp


# ----------------------------- CLI ----------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Run folder inference on GeoTIFF tiles.")
    ap.add_argument("--ckpt", type=str, default="", help="(optional) path to checkpoint .pt")
    ap.add_argument("--in", dest="in_dir", type=str, required=True, help="folder containing input *.tif tiles")
    ap.add_argument("--out", dest="out_dir", type=str, required=True, help="folder to write outputs")
    ap.add_argument("--threshold", type=float, default=0.45, help="probability threshold for mask")
    ap.add_argument("--encoder", type=str, default="resnet18", help="encoder backbone name")
    args = ap.parse_args()

    device = "cpu"  # GitHub runners are CPU-only
    fix_seeds(42)

    model = load_unet(args.ckpt if args.ckpt else None,
                      encoder_name=args.encoder,
                      device=device)

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(in_dir.glob("*.tif"))
    if not tifs:
        print(f"[warn] no .tif files found under: {in_dir.resolve()}")
        return

    print(f"[info] found {len(tifs)} tiles; writing outputs to: {out_dir.resolve()}")
    for tif in tifs:
        proba_fp, mask_fp = infer_tile(model, tif, out_dir, threshold=args.threshold, device=device)
        print(f"[ok] {tif.name} -> {proba_fp.name}, {mask_fp.name}")


if __name__ == "__main__":
    main()
