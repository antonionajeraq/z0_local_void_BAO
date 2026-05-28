import numpy as np
import sys
import csv

# Check that the user provided exactly one argument (rec or drag).
if len(sys.argv) != 2:
    print("Usage: python3 CODE.py <rec|drag>")
    sys.exit(1)

Mode = sys.argv[1].lower()  # Make lowercase for flexibility.

#Load parameter ranges.
with open("Parameter_ranges.csv", "r", newline="") as f:
    reader = csv.reader(f)
    (wbt_min, wbt_max, N_wbt), \
    (wbct_min, wbct_max, N_wbct) = reader

wbt_min, wbt_max = float(wbt_min), float(wbt_max)
wbct_min, wbct_max = float(wbct_min), float(wbct_max)
N_wbt = int(N_wbt)
N_wbct = int(N_wbct)
N = N_wbt * N_wbct

#Prepare 2D T_values array.
T_values = np.zeros((N_wbt, N_wbct))

# Determine which dataset to use
if Mode == "rec":
    Epoch = "T_star"
elif Mode == "drag":
    Epoch = "T_d"
else:
    raise ValueError("Mode must be either 'rec' or 'drag'.")

#Load full 2D array directly (no z0 loop).
T_values[:, :] = np.loadtxt(f"{Epoch}/{Epoch}_values_FIRAS.csv", delimiter=",")

# Grid values.
wbt_values = np.linspace(wbt_min, wbt_max, N_wbt)
wbct_values = np.linspace(wbct_min, wbct_max, N_wbct)

# Reference values.
wbt_ref = 0.5*(wbt_min + wbt_max)
wbct_ref = 0.5*(wbct_min + wbct_max)

# Load reference, gradient, Hessian.
wbt_ref_loaded, wbct_ref_loaded, T_ref = np.loadtxt(f"{Epoch}/{Epoch}_ref.csv", delimiter=",")
assert np.isclose(wbt_ref_loaded, wbt_ref, atol=1e-9, rtol=0)
assert np.isclose(wbct_ref_loaded, wbct_ref, atol=1e-9, rtol=0)

Gradient = np.loadtxt(f"{Epoch}/{Epoch}_gradient.csv", delimiter=",").ravel()
assert Gradient.shape[0] == 2, f"Gradient should have length 2, got {Gradient.shape[0]}"

Hessian = np.loadtxt(f"{Epoch}/{Epoch}_Hessian.csv", delimiter=",")
Hessian = np.atleast_2d(Hessian)
assert Hessian.shape == (2, 2), f"Hessian should be 2x2, got {Hessian.shape}"

# Compute max and RMS errors.
Max_abs_T_error = -1.0
Sum_error_sq = 0.0
for i in range(N_wbt):
    wbt = wbt_values[i]
    for j in range(N_wbct):
        wbct = wbct_values[j]
        d = np.array([wbt - wbt_ref, wbct - wbct_ref])
        T_CLASS = T_values[i, j]
        T_fit = T_ref + Gradient @ d + 0.5 * (d @ Hessian @ d)
        T_error = T_fit - T_CLASS
        Sum_error_sq += T_error ** 2
        Max_abs_T_error = max(Max_abs_T_error, abs(T_error))
RMS_T_error = np.sqrt(Sum_error_sq/N)

# Save fitting error.
np.savetxt(f"{Epoch}/{Epoch}_error.csv", np.array([[Max_abs_T_error, RMS_T_error]]), fmt="%.10f", delimiter=",")
