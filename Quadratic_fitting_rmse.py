import numpy as np
import sys
import csv

# Check that the user provided exactly one argument (rec or drag).
if len(sys.argv) != 2:
    print("Usage: python3 CODE.py <rec|drag>")
    sys.exit(1)

Mode = sys.argv[1].lower() #Make lowercase for flexibility.

with open("Parameter_ranges.csv", "r", newline="") as f:
    reader = csv.reader(f)
    (wbt_min, wbt_max, N_wbt), \
    (wbct_min, wbct_max, N_wbct), \
    (z0_min, z0_max, N_z0) = reader

wbt_min, wbt_max = float(wbt_min), float(wbt_max)
wbct_min, wbct_max = float(wbct_min), float(wbct_max)
z0_min, z0_max = float(z0_min), float(z0_max)

N_wbt = int(N_wbt)
N_wbct = int(N_wbct)
N_z0 = int(N_z0)
N = N_wbt*N_wbct*N_z0

T_values = np.zeros((N_wbt, N_wbct, N_z0))
if Mode == "rec":
    folder = "T_star"
elif Mode == "drag":
    folder = "T_d"
else:
    raise ValueError("Mode must be either 'rec' or 'drag'.")

for k in range(N_z0): T_values[:, :, k] = np.loadtxt(f"{folder}/{folder}_values_z0_{k}.csv", delimiter=",")

wbt_values = np.linspace(wbt_min, wbt_max, N_wbt)
wbct_values = np.linspace(wbct_min, wbct_max, N_wbct)
z0_values = np.linspace(z0_min, z0_max, N_z0)

wbt_ref_expected = 0.5*(wbt_min + wbt_max)
wbct_ref_expected = 0.5*(wbct_min + wbct_max)
z0_ref_expected = 0.5*(z0_min + z0_max)

# Load reference temperature and where this is valid.
wbt_ref, wbct_ref, z0_ref, T_ref = np.loadtxt(f"{folder}/{folder}_ref.csv", delimiter=",")
assert np.isclose(wbt_ref, wbt_ref_expected, atol=1e-9, rtol=0)
assert np.isclose(wbct_ref, wbct_ref_expected, atol=1e-9, rtol=0)
assert np.isclose(z0_ref, z0_ref_expected, atol=1e-9, rtol=0)

# Load Gradient.
Gradient = np.loadtxt(f"{folder}/{folder}_gradient.csv", delimiter=",")
Gradient = np.atleast_1d(Gradient).ravel()  # ensure 1D
assert Gradient.shape[0] == 3, f"Gradient should have length 3, got {Gradient.shape[0]}"

# Load Hessian.
Hessian = np.loadtxt(f"{folder}/{folder}_Hessian.csv", delimiter=",")
Hessian = np.atleast_2d(Hessian)  # ensure 2D
assert Hessian.shape == (3, 3), f"Hessian should be 3x3, got {Hessian.shape}"

Max_abs_T_error = -1.0
Sum_error_sq = 0.0
for i in range(N_wbt):
    wbt = wbt_values[i]

    for j in range(N_wbct):
        wbct = wbct_values[j]

        for k in range(N_z0):
            z0 = z0_values[k]
            d = np.array([wbt - wbt_ref, wbct - wbct_ref, z0 - z0_ref])
            T_CLASS = T_values[i, j, k]
            T_fit = T_ref + Gradient@d + 0.5*(d@Hessian@d)
            T_error = T_fit - T_CLASS
            Sum_error_sq += T_error*T_error
            Max_abs_T_error = max(Max_abs_T_error, abs(T_error)) #Use np.abs for arrays.
        #end
    #end
#end

RMS_T_error = np.sqrt(Sum_error_sq/N)
np.savetxt(f"{folder}/{folder}_error.csv", np.array([[Max_abs_T_error, RMS_T_error]]), fmt="%.10f", delimiter=",")
