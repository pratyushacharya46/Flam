# Parametric Curve Fitting — R&D / AI Assignment

This project estimates the unknown parameters θ, M, and X for the parametric curve:

x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X  
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)  
for 6 < t < 60.

The dataset contains ~1500 measured (x, y) points sampled from this curve. The goal is to recover the original parameter values.

---

## Final Estimated Parameters

θ (radians): 0.5235983  
θ (degrees): ≈ 30°  
M: 0.03  
X: 55  

---

## Final Submission Expression (as required)

(t*cos(0.5235983) - exp(0.03*abs(t))*sin(0.3*t)*sin(0.5235983) + 55, 42 + t*sin(0.5235983) + exp(0.03*abs(t))*sin(0.3*t)*cos(0.5235983))

Domain: 6 ≤ t ≤ 60

---

## Method Summary

1. Applied PCA on the dataset to estimate the main orientation, providing an initial value for θ.
2. Shifted and rotated the coordinates to isolate the wobble behavior in a (u, v) space.
3. Estimated M by fitting the amplitude envelope of v with a log-linear model.
4. Used Nelder–Mead optimization to minimize L1 curve-to-curve distance and refine θ, M, and X together.

This approach directly addresses the evaluation metric described in the assignment.

---

## Visual Results

Data overlay with fitted curve:  
plots/overlay.png

Wobble in local coordinate frame:  
plots/local_frame.png

---

## How to Reproduce

python -m venv .venv  
. .venv/Scripts/activate   (on Windows)  
pip install -r requirements.txt  
python -m src.fit --data data/xy_data.csv --N 500 --seed 42

Outputs generated:  
results/params.json  
plots/overlay.png  
plots/local_frame.png  

---

## Repository Structure

data/xy_data.csv  
src/fit.py  
src/geometry.py  
src/model.py  
src/losses.py  
plots/  
results/  
README.md  

---

## References

[1] I. T. Jolliffe, Principal Component Analysis, 2nd ed., Springer, 2002.  
[2] J. A. Nelder and R. Mead, “A simplex method for function minimization,” The Computer Journal, 1965.  
[3] C. R. Harris et al., “Array programming with NumPy,” Nature, 2020.  
[4] P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” Nature Methods, 2020.
