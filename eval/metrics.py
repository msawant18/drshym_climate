import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    return tp, fp, fn, tn


def iou(y_true, y_pred):
    tp, fp, fn, _ = confusion(y_true, y_pred)
    denom = tp + fp + fn
    return float(tp / (denom + 1e-6))


def precision(y_true, y_pred):
    tp, fp, _, _ = confusion(y_true, y_pred)
    return float(tp / (tp + fp + 1e-6))


def recall(y_true, y_pred):
    tp, _, fn, _ = confusion(y_true, y_pred)
    return float(tp / (tp + fn + 1e-6))


def f1(y_true, y_pred):
    p = precision(y_true, y_pred); r = recall(y_true, y_pred)
    return float(2*p*r / (p + r + 1e-6))


def brier(y_true, p):
    y = y_true.astype("float32"); p = p.astype("float32")
    return float(((p - y)**2).mean())


def ece(y_true, p, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece_val = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if m.sum() == 0: continue
        acc = (y_true[m] > 0.5).mean()
        conf = p[m].mean()
        ece_val += (m.mean()) * abs(acc - conf)
    return float(ece_val)
