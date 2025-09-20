from pathlib import Path
from typing import Dict

import numpy as np
import torch
import rasterio as rio

from models.encoder_backbones import make_encoder
from models.unet import UNet


def load_model(ckpt_path: str = None, encoder: str = "resnet18", device: str = None) -> torch.nn.Module:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    layers, chs = make_encoder(encoder, pretrained=False)
    model = UNet(layers, chs).to(dev)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state)
    model.eval()
    return model


def infer_tile(model: torch.nn.Module, tile_tif: str, threshold: float = 0.5) -> Dict[str, str]:
    dev = next(model.parameters()).device
    with rio.open(tile_tif) as src:
        arr = src.read(1).astype("float32")
        x = torch.from_numpy(arr)[None, None].to(dev)
        with torch.inference_mode():
            logits = model(x)
            proba = torch.sigmoid(logits).squeeze().cpu().numpy().astype("float32")
        mask = (proba >= threshold).astype("uint8")
        profile = src.profile.copy(); profile.update(count=1, dtype="float32")
    proba_path = str(Path(tile_tif).with_suffix(".proba.tif"))
    mask_path = str(Path(tile_tif).with_suffix(".mask.tif"))
    with rio.open(proba_path, "w", **profile) as dst:
        dst.write(proba, 1)
    profile["dtype"] = "uint8"
    with rio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
    return {"proba": proba_path, "mask": mask_path}
