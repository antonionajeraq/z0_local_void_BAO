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
    (wbct_min, wbct_max, N_wbct) = reader

wbt_min, wbt_max = float(wbt_min), float(wbt_max)
wbct_min, wbct_max = float(wbct_min), float(wbct_max)

N_wbt = int(N_wbt)
N_wbct = int(N_wbct)

T_values = np.zeros((N_wbt, N_wbct))
if Mode == "rec":
    Epoch = "T_star"
elif Mode == "drag":
    Epoch = "T_d"
else:
    raise ValueError("Mode must be either 'rec' or 'drag'.")

T_values[:, :] = np.loadtxt(f"{Epoch}/{Epoch}_values_FIRAS.csv", delimiter=",")

wbt_values = np.linspace(wbt_min, wbt_max, N_wbt)
wbct_values = np.linspace(wbct_min, wbct_max, N_wbct)

wbt_ref = 0.5*(wbt_min + wbt_max)
wbct_ref = 0.5*(wbct_min + wbct_max)

N = N_wbt*N_wbct
f_values = np.empty(N)
x_values = np.empty(N)
y_values = np.empty(N)

w = -1
for i in range(N_wbt):
    x = wbt_values[i]

    for j in range(N_wbct):
        y = wbct_values[j]
        w += 1
        f_values[w] = T_values[i, j]
        x_values[w] = x
        y_values[w] = y
    #end
#end

x_values -= wbt_ref
y_values -= wbct_ref

xx_values = x_values*x_values
yy_values = y_values*y_values
xy_values = x_values*y_values

fx_values = f_values*x_values
fy_values = f_values*y_values

b_x = np.sum(fx_values)/np.sum(xx_values)
b_y = np.sum(fy_values)/np.sum(yy_values)
c_xy = np.sum(f_values*xy_values)/np.sum(xy_values*xy_values)

S_xx = np.sum(xx_values)
S_yy = np.sum(yy_values)

S_xxxx = np.sum(xx_values*xx_values)
S_yyyy = np.sum(yy_values*yy_values)
S_xxyy = np.sum(xx_values*yy_values)

S_f = np.sum(f_values)
S_fxx = np.sum(f_values*xx_values)
S_fyy = np.sum(f_values*yy_values)

A = np.zeros((3, 3))
A[0, 0] = float(N)
A[0, 1] = S_xx
A[0, 2] = S_yy
A[1, 1] = S_xxxx
A[1, 2] = S_xxyy
A[2, 2] = S_yyyy
i, j = np.triu_indices_from(A, k=1)
A[j, i] = A[i, j]

v = np.array([S_f, S_fxx, S_fyy])
f_0, c_xx, c_yy = np.linalg.solve(A, v)

Gradient = np.array([b_x, b_y])

Hessian = np.zeros((2, 2))
Hessian[0, 0] = c_xx
Hessian[0, 1] = c_xy
Hessian[1, 1] = c_yy
Hessian = Hessian + Hessian.T

np.savetxt(f"{Epoch}/{Epoch}_ref.csv", np.array([[wbt_ref, wbct_ref, f_0]]), fmt="%.10f", delimiter=",")
np.savetxt(f"{Epoch}/{Epoch}_gradient.csv", Gradient.reshape(1, -1), fmt="%.10f", delimiter=",")
np.savetxt(f"{Epoch}/{Epoch}_Hessian.csv", Hessian, fmt="%.10f", delimiter=",")

#Estimated f = f_0 + Gradient.d + 0.5*d.T*Hessian*d, where d is displacement from (x_0, y_0).
