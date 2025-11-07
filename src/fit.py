import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .geometry import pca_theta, to_local, clamp_theta, clamp_M, clamp_X
from .model import wobble, predict_curve
from .losses import loss_local, loss_curve

import matplotlib.pyplot as plt


def init_params(points_xy: np.ndarray):
    # Theta via PCA
    theta0 = pca_theta(points_xy)
    theta0 = clamp_theta(theta0)

    # X via mean alignment around t mid ~ (6+60)/2 = 33
    t_mid = 33.0
    X0 = float(points_xy[:, 0].mean() - t_mid * np.cos(theta0))
    X0 = clamp_X(X0)

    # M via envelope regression in local frame
    uv = to_local(points_xy, theta0, X0)
    u, v = uv[:, 0], uv[:, 1]
    s = np.sin(0.3 * u)

    # Avoid division near sin ≈ 0
    mask = np.abs(s) > 0.2
    if mask.sum() >= 10:
        a = np.abs(v[mask]) / (np.abs(s[mask]) + 1e-8)
        a = np.clip(a, 1e-8, 1e6)
        M0 = float(np.polyfit(np.abs(u[mask]), np.log(a), 1)[0])
    else:
        M0 = 0.0

    M0 = clamp_M(M0)
    return theta0, M0, X0


def grid_search(
    points_xy: np.ndarray,
    theta0: float,
    M0: float,
    X0: float,
    n_theta=17,
    n_M=17,
    n_X=17,
    span_deg=10.0,
    X_span=20.0,
):
    thetas = np.deg2rad(
        np.linspace(
            np.rad2deg(theta0) - span_deg, np.rad2deg(theta0) + span_deg, n_theta
        )
    )
    thetas = np.clip(thetas, 0.0 + 1e-4, np.deg2rad(50.0) - 1e-4)

    Ms = np.linspace(max(-0.05 + 1e-6, M0 - 0.02), min(0.05 - 1e-6, M0 + 0.02), n_M)
    Xs = np.linspace(max(0.0 + 1e-6, X0 - X_span), min(100.0 - 1e-6, X0 + X_span), n_X)

    best = []
    for th in thetas:
        for Mm in Ms:
            for Xv in Xs:
                L = loss_local(points_xy, th, Mm, Xv)
                best.append((L, th, Mm, Xv))

    best.sort(key=lambda x: x[0])
    return best[:8]  # top seeds


def refine(points_xy: np.ndarray, seeds, N_curve=500):
    bounds = [
        (0.0 + 1e-4, np.deg2rad(50.0) - 1e-4),
        (-0.05 + 1e-6, 0.05 - 1e-6),
        (0.0 + 1e-6, 100.0 - 1e-6),
    ]

    def _obj(x):
        th, Mm, Xv = x
        return loss_local(points_xy, th, Mm, Xv)

    candidates = []
    for L0, th0, M0, X0 in seeds:
        res = minimize(
            _obj,
            x0=np.array([th0, M0, X0]),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
        )
        th, Mm, Xv = res.x
        th, Mm, Xv = clamp_theta(th), clamp_M(Mm), clamp_X(Xv)

        L_local = loss_local(points_xy, th, Mm, Xv)
        L_curve = loss_curve(points_xy, th, Mm, Xv, N=N_curve)

        candidates.append((L_curve, L_local, th, Mm, Xv))

    candidates.sort(key=lambda z: z[0])
    return candidates[0], candidates


def make_plots(points_xy: np.ndarray, theta: float, M: float, X: float, N: int = 500):
    os.makedirs("plots", exist_ok=True)

    t = np.linspace(6.0, 60.0, N)
    pred = predict_curve(t, theta, M, X)

    # Overlay
    plt.figure()
    plt.scatter(points_xy[:, 0], points_xy[:, 1], s=6, alpha=0.6, label="data")
    plt.plot(pred[:, 0], pred[:, 1], lw=2, label="model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Data vs Predicted Curve")
    plt.tight_layout()
    plt.savefig("plots/overlay.png", dpi=180)
    plt.close()

    # Local frame wobble
    uv = to_local(points_xy, theta, X)
    u, v = uv[:, 0], uv[:, 1]
    o = np.argsort(u)
    u_s, v_s = u[o], v[o]

    plt.figure()
    plt.plot(u_s, v_s, lw=1, label="data v(u)")
    plt.plot(u_s, wobble(u_s, M), lw=2, label="model wobble")
    plt.xlabel("u (~ t)")
    plt.ylabel("v")
    plt.legend()
    plt.title("Local Frame Wobble Fit")
    plt.tight_layout()
    plt.savefig("plots/local_frame.png", dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="CSV with x,y")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    pts = df[["x", "y"]].to_numpy(dtype=float)

    theta0, M0, X0 = init_params(pts)
    seeds = grid_search(pts, theta0, M0, X0)
    (Lcurve_best, Llocal_best, theta, M, X), all_cands = refine(
        pts, seeds, N_curve=args.N
    )

    os.makedirs("results", exist_ok=True)
    out = {
        "theta_rad": float(theta),
        "theta_deg": float(np.rad2deg(theta)),
        "M": float(M),
        "X": float(X),
        "scores": {"L1_curve": float(Lcurve_best), "L1_local": float(Llocal_best)},
    }
    with open("results/params.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nFinal parameters:")
    print(json.dumps(out, indent=2))

    print("\nPaste this into Desmos / submission:")
    print(
        f"(t*cos({theta}) - exp({M}*abs(t))*sin(0.3*t)*sin({theta}) + {X}, "
        f"42 + t*sin({theta}) + exp({M}*abs(t))*sin(0.3*t)*cos({theta}))"
    )

    make_plots(pts, theta, M, X, N=args.N)


if __name__ == "__main__":
    main()
