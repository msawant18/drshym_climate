import numpy as np

class TemperatureScaler:
    def __init__(self, T: float = 1.0):
        self.T = T

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr=0.01, iters=500):
        T = self.T
        for _ in range(iters):
            p = 1/(1+np.exp(-logits/T))
            grad = np.mean((p - labels) * logits * (p*(1-p)))/(T**2 + 1e-6)
            T -= lr*grad
            T = float(max(0.05, min(10.0, T)))
        self.T = T
        return T

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-logits/self.T))
