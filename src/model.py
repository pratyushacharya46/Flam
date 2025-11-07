import numpy as np


def wobble(u: np.ndarray, M: float) -> np.ndarray:
    """
    W(t) = e^{M|t|} * sin(0.3t)
    """
    return np.exp(M * np.abs(u)) * np.sin(0.3 * u)


def predict_curve(t: np.ndarray, theta: float, M: float, X: float) -> np.ndarray:
    """
    Parametric curve:
      x(t) = t*cosθ - wobble(t)*sinθ + X
      y(t) = 42 + t*sinθ + wobble(t)*cosθ
    """
    w = wobble(t, M)
    c, s = np.cos(theta), np.sin(theta)

    x = t * c - w * s + X
    y = 42.0 + t * s + w * c

    return np.stack([x, y], axis=1)
