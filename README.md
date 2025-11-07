# Parametric Curve Fitting — R&D / AI Assignment

This project estimates the unknown parameters \(\theta\), \(M\), and \(X\) in the curve:

\[
\begin{aligned}
x(t) &= t\cos\theta - e^{M|t|}\sin(0.3t)\sin\theta + X, \\
y(t) &= 42 + t\sin\theta + e^{M|t|}\sin(0.3t)\cos\theta,
\end{aligned}
\quad 6 < t < 60.
\]

## Final Estimated Parameters
| Parameter | Value |
|----------|-------|
| \(\theta\) (radians) | **0.5235983** |
| \(\theta\) (degrees) | **≈ 30°** |
| \(M\) | **0.03** |
| \(X\) | **55** |

## Final Submission Expression (Paste into Report / Desmos)

(tcos(0.5235983) - exp(0.03abs(t))sin(0.3t)sin(0.5235983) + 55,
42 + tsin(0.5235983) + exp(0.03*abs(t))sin(0.3t)*cos(0.5235983))


## How to Run
```bash
python -m venv .venv
. .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python -m src.fit --data data/xy_data.csv --N 500 --seed 42


After running, results are stored in:
results/params.json
plots/overlay.png
plots/local_frame.png

Method Summary

1. PCA estimates initial curve angle.

2. Data is transformed into local coordinates (u,v).

3. Exponential wobble envelope is fitted to estimate M.

4. Nonlinear optimization (Nelder–Mead) minimizes L1 curve distance.

5. Final values are selected based on curve-to-curve error scoring.

IEEE References

[1] I. T. Jolliffe, Principal Component Analysis, Springer, 2002.
[2] J. A. Nelder and R. Mead, “A simplex method for function minimization,” The Computer Journal, 1965.
[3] C. R. Harris et al., “Array programming with NumPy,” Nature, 2020.
[4] P. Virtanen et al., “SciPy 1.0,” Nature Methods, 2020.