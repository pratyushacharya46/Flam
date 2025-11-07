# Parametric Curve Fitting for R&D / AI Assignment

This repository implements parameter estimation for a parametric curve defined by:

\[
\begin{aligned}
x(t) &= t\cos\theta - e^{M|t|}\sin(0.3t)\sin\theta + X, \\
y(t) &= 42 + t\sin\theta + e^{M|t|}\sin(0.3t)\cos\theta,
\end{aligned}
\quad 6 < t < 60.
\]

The task is to determine the unknown parameters \(\theta\), \(M\), and \(X\) from a set of 1500 measured \((x, y)\) points.

---

## Final Estimated Parameters

| Parameter | Value | Description |
|----------|-------|-------------|
| \(\theta\) (radians) | **0.5235983** | Rotation angle |
| \(\theta\) (degrees) | **≈ 30°** | Angle in degrees |
| \(M\) | **0.03** | Exponential growth factor |
| \(X\) | **55** | Horizontal translation |

---

## Final Submission Expression (Copy to Report / Desmos)



(tcos(0.5235983) - exp(0.03abs(t))sin(0.3t)sin(0.5235983) + 55,
42 + tsin(0.5235983) + exp(0.03*abs(t))sin(0.3t)*cos(0.5235983))


---

## Visualization

### Model vs Data
![Overlay](plots/overlay.png)

### Wobble Fit in Local Coordinate Frame
![Local Frame](plots/local_frame.png)

---

## Approach

1. **Initial Rotation Estimate (θ):**  
   Principal Component Analysis (PCA) is applied to the point cloud to estimate the dominant direction, which provides an initial approximation for \(\theta\).

2. **Coordinate Transformation:**  
   The points are shifted by \([X, 42]\) and rotated into the local \((u, v)\) coordinate frame where \(u \approx t\) and \(v \approx e^{M|t|}\sin(0.3t)\).

3. **Exponential Envelope Fitting (M):**  
   The amplitude envelope \(|v| / |\sin(0.3u)|\) is analyzed to estimate \(M\) using a log-linear regression.

4. **Joint Parameter Optimization:**  
   A Nelder–Mead optimization minimizes the L1 distance between the predicted curve and the measured data.  
   The final score is computed by uniform sampling of \(t \in [6, 60]\) and comparing curve-to-curve L1 distance.

---

## How to Run

```bash
python -m venv .venv
. .venv/Scripts/activate        # (Windows)
pip install -r requirements.txt
python -m src.fit --data data/xy_data.csv --N 500 --seed 42

## Output files:

results/params.json     # final parameter values
plots/overlay.png       # curve vs data visualization
plots/local_frame.png   # wobble fit visualization

Repository Structure

├── data/
│   └── xy_data.csv
├── src/
│   ├── fit.py
│   ├── geometry.py
│   ├── model.py
│   └── losses.py
├── plots/
├── results/
├── requirements.txt
└── README.md

References (IEEE Format)

[1] I. T. Jolliffe, Principal Component Analysis, 2nd ed. Springer, 2002.
[2] J. A. Nelder and R. Mead, “A simplex method for function minimization,” The Computer Journal, 1965.
[3] C. R. Harris et al., “Array programming with NumPy,” Nature, 2020.
[4] P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” Nature Methods, 2020.

