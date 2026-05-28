import numpy as np
import sys
import csv

# Check that the user provided exactly one argument (rec or drag).
if len(sys.argv) != 2:
    print("Usage: python3 CODE.py <rec|drag>")
    sys.exit(1)

Mode = sys.argv[1].lower()  # make lowercase for flexibility.

#def Function_evaluation(x, y, z):
#    return (3.0
#        + 2.0*x - 1.0*y + 4.0*z
#        + 5.0*x*x + 6.0*y*y + 7.0*z*z
#        + 8.0*x*y + 9.0*x*z + 10.0*y*z)

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

wbt_ref = 0.5*(wbt_min + wbt_max)
wbct_ref = 0.5*(wbct_min + wbct_max)
z0_ref = 0.5*(z0_min + z0_max)

N = N_wbt*N_wbct*N_z0
f_values = np.empty(N)
x_values = np.empty(N)
y_values = np.empty(N)
z_values = np.empty(N)

w = -1
for i in range(N_wbt):
    x = wbt_values[i]

    for j in range(N_wbct):
        y = wbct_values[j]

        for k in range(N_z0):
            z = z0_values[k]
            w += 1
            f_values[w] = T_values[i, j, k]
            x_values[w] = x
            y_values[w] = y
            z_values[w] = z
        #end
    #end
#end

x_values -= wbt_ref
y_values -= wbct_ref
z_values -= z0_ref

xx_values = x_values*x_values
yy_values = y_values*y_values
zz_values = z_values*z_values

xy_values = x_values*y_values
xz_values = x_values*z_values
yz_values = y_values*z_values

fx_values = f_values*x_values
fy_values = f_values*y_values
fz_values = f_values*z_values

b_x = np.sum(fx_values)/np.sum(xx_values)
b_y = np.sum(fy_values)/np.sum(yy_values)
b_z = np.sum(fz_values)/np.sum(zz_values)

c_xy = np.sum(f_values*xy_values)/np.sum(xy_values*xy_values)
c_xz = np.sum(f_values*xz_values)/np.sum(xz_values*xz_values)
c_yz = np.sum(f_values*yz_values)/np.sum(yz_values*yz_values)

S_xx = np.sum(xx_values)
S_yy = np.sum(yy_values)
S_zz = np.sum(zz_values)

S_xxxx = np.sum(xx_values*xx_values)
S_yyyy = np.sum(yy_values*yy_values)
S_zzzz = np.sum(zz_values*zz_values)

S_xxyy = np.sum(xx_values*yy_values)
S_xxzz = np.sum(xx_values*zz_values)
S_yyzz = np.sum(yy_values*zz_values)

S_f = np.sum(f_values)
S_fxx = np.sum(f_values*xx_values)
S_fyy = np.sum(f_values*yy_values)
S_fzz = np.sum(f_values*zz_values)

A = np.zeros((4, 4))
A[0, 0] = float(N)
A[0, 1] = S_xx
A[0, 2] = S_yy
A[0, 3] = S_zz
A[1, 1] = S_xxxx
A[1, 2] = S_xxyy
A[1, 3] = S_xxzz
A[2, 2] = S_yyyy
A[2, 3] = S_yyzz
A[3, 3] = S_zzzz
i, j = np.triu_indices_from(A, k=1)
A[j, i] = A[i, j]

v = np.array([S_f, S_fxx, S_fyy, S_fzz])
f_0, c_xx, c_yy, c_zz = np.linalg.solve(A, v)

Gradient = np.array([b_x, b_y, b_z])

Hessian = np.zeros((3, 3))
Hessian[0, 0] = c_xx
Hessian[0, 1] = c_xy
Hessian[0, 2] = c_xz
Hessian[1, 1] = c_yy
Hessian[1, 2] = c_yz
Hessian[2, 2] = c_zz
Hessian = Hessian + Hessian.T

np.savetxt(f"{folder}/{folder}_ref.csv", np.array([[wbt_ref, wbct_ref, z0_ref, f_0]]), fmt="%.10f", delimiter=",")
np.savetxt(f"{folder}/{folder}_gradient.csv", Gradient.reshape(1, -1), fmt="%.10f", delimiter=",")
np.savetxt(f"{folder}/{folder}_Hessian.csv", Hessian, fmt="%.10f", delimiter=",")

#Estimated f = f_0 + Gradient.d + 0.5*d.T*Hessian*d, where d is displacement from (x_0, y_0, z_0).


#print("Recovered f_0:", f_0)
#print("Recovered Gradient:", Gradient)
#print("Recovered Hessian:\n", Hessian)

#print("\nExact f_0: 3.0")
#print("Exact Gradient: [2.0, -1.0, 4.0]")
#print("Exact Hessian:\n",
#      np.array([[10.0, 8.0, 9.0],
#                [8.0, 12.0, 10.0],
#                [9.0, 10.0, 14.0]]))
