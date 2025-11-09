# Parametric Curve Fitting — R&D / AI Assignment

This project estimates the unknown parameters θ, M, and X for the parametric curve:

x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X  
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)  
for 6 < t < 60.

The dataset contains ~1500 measured (x, y) points sampled from this curve. The goal is to recover the original parameter values as accurately as possible.

---

## Final Estimated Parameters

θ (radians): **0.5235983**  
θ (degrees): **≈ 30°**  
M: **0.03**  
X: **55**

---

## Final Submission Expression 

(t*cos(0.5235983) - exp(0.03*abs(t))*sin(0.3*t)*sin(0.5235983) + 55,  
 42 + t*sin(0.5235983) + exp(0.03*abs(t))*sin(0.3*t)*cos(0.5235983))

**Domain:** 6 ≤ t ≤ 60

---

## Method Summary

1. Applied PCA to estimate the primary direction of the curve, giving an initial approximation for θ.
2. Translated and rotated the dataset into a (u, v) local coordinate system to isolate the oscillatory component.
3. Estimated M by fitting the exponential envelope of v using a log-linear relationship.
4. Performed Nelder–Mead optimization to jointly refine θ, M, and X by minimizing L1 curve-to-curve distance.

This procedure directly aligns with the scoring criteria provided in the assignment.

---

## Visual Results

### Data vs. Fitted Curve  
![Data vs Curve](plots/overlay.png)

### Wobble Behavior in Local Coordinates  
![Wobble Local Frame](plots/local_frame.png)

---

## How to Reproduce

python -m venv .venv  
. .venv/Scripts/activate      
pip install -r requirements.txt  
python -m src.fit --data data/xy_data.csv --N 500 --seed 42

Generated output files:  
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
requirements.txt  
README.md

---

## References

[1] I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer, 2002.  
[2] J. A. Nelder and R. Mead, “A simplex method for function minimization,” *The Computer Journal*, 1965.  
[3] C. R. Harris et al., “Array programming with NumPy,” *Nature*, 2020.  
[4] P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” *Nature Methods*, 2020.
