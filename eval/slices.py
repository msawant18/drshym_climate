import numpy as np
from typing import Dict
from eval.metrics import iou


def by_intensity_deciles(img: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Dict[str, float]:
    qs = np.quantile(img, np.linspace(0,1,k+1))
    out = {}
    for i in range(k):
        m = (img >= qs[i]) & (img < qs[i+1])
        if m.sum() < 10: continue
        out[f"q{i}"] = iou(y_true[m], y_pred[m])
    return out
