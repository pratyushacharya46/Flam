import numpy as np
from .geometry import to_local, resample_polyline
from .model import wobble, predict_curve


def loss_local(points_xy: np.ndarray, theta: float, M: float, X: float) -> float:
    """
    Fast robust loss in local (u,v) frame:
       v ≈ wobble(u)
    Uses median L1 to stabilize against noise.
    """
    uv = to_local(points_xy, theta, X)
    u, v = uv[:, 0], uv[:, 1]
    pred = wobble(u, M)
    return float(np.median(np.abs(v - pred)))


def loss_curve(
    points_xy: np.ndarray, theta: float, M: float, X: float, N: int = 500
) -> float:
    """
    Rubric-aligned loss:
      1. Convert data to local (u,v)
      2. Sort by u to get traversal order
      3. Resample expected curve (arc-length uniform)
      4. Compare to predicted parametric model at uniform t ∈ [6,60]
      5. Return mean L1 distance
    """
    # Sort points along curve progression
    uv = to_local(points_xy, theta, X)
    idx = np.argsort(uv[:, 0])
    ordered = points_xy[idx]

    # Resample expected curve
    exp_curve = resample_polyline(ordered, N)

    # Predict model curve
    t = np.linspace(6.0, 60.0, N)
    pred = predict_curve(t, theta, M, X)

    return float(np.mean(np.abs(pred - exp_curve)))
