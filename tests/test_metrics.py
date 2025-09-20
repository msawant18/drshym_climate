import numpy as np
from eval.metrics import iou, f1

def test_metrics_toy():
    y = np.array([[1,0],[1,0]], dtype=np.uint8)
    p = np.array([[1,0],[0,0]], dtype=np.uint8)
    assert 0 <= iou(y,p) <= 1
    assert 0 <= f1(y,p) <= 1
