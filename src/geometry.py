import numpy as np
from sklearn.decomposition import PCA

EPS = 1e-8


# Rotation matrix R(theta)
def rot_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


# Convert global coordinates → local (u,v)
def to_local(points_xy: np.ndarray, theta: float, X: float) -> np.ndarray:
    shifted = points_xy - np.array([X, 42.0])
    Rm = rot_mat(-theta)
    return shifted @ Rm.T


# PCA to estimate direction → initial θ
def pca_theta(points_xy: np.ndarray) -> float:
    pca = PCA(n_components=2)
    pca.fit(points_xy)
    v = pca.components_[0]
    return float(np.arctan2(v[1], v[0]))  # radians


# Resample a polyline to N points by arc length
def resample_polyline(poly: np.ndarray, N: int) -> np.ndarray:
    if len(poly) < 2:
        return np.repeat(poly, N, axis=0)

    seg = np.diff(poly, axis=0)
    d = np.sqrt((seg**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]

    if L < EPS:
        return np.repeat(poly[:1], N, axis=0)

    targets = np.linspace(0.0, L, N)
    out = np.zeros((N, 2), dtype=float)

    j = 0
    for i, tt in enumerate(targets):
        while j + 1 < len(s) and s[j + 1] < tt:
            j += 1
        if j + 1 >= len(s):
            out[i] = poly[-1]
        else:
            denom = s[j + 1] - s[j]
            t = 0.0 if denom < EPS else (tt - s[j]) / denom
            out[i] = poly[j] * (1 - t) + poly[j + 1] * t

    return out


# Bounds
def clamp_theta(theta: float) -> float:
    theta = np.mod(theta, 2 * np.pi)
    if theta > np.pi:
        theta -= np.pi
    return float(np.clip(theta, 0.0 + 1e-4, np.deg2rad(50.0) - 1e-4))


def clamp_M(M: float) -> float:
    return float(np.clip(M, -0.05 + 1e-6, 0.05 - 1e-6))


def clamp_X(X: float) -> float:
    return float(np.clip(X, 0.0 + 1e-6, 100.0 - 1e-6))
