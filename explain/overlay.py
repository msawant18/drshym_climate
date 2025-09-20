from pathlib import Path
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt


def overlay_heatmap(scene_tif: str, proba_tif: str) -> str:
    with rio.open(scene_tif) as src:
        bg = src.read(1)
    with rio.open(proba_tif) as pr:
        proba = pr.read(1)
    bg_disp = (bg - np.percentile(bg,1))/(np.percentile(bg,99)-np.percentile(bg,1)+1e-6)
    bg_disp = np.clip(bg_disp,0,1)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(bg_disp, cmap='gray')
    plt.imshow(proba, cmap='jet', alpha=0.4, vmin=0, vmax=1)
    plt.axis('off')
    out = str(Path(proba_tif).with_suffix('.overlay.png'))
    fig.savefig(out, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return out
