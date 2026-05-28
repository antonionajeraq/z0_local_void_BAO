import numpy as np
from classy import Class
import csv
import os

cosmo = Class()
T_FIRAS = 2.72548 #Fixsen_2009.

# Base parameters, including T_CMB.
base_params = {
    'output': '',
    'lensing': 'no',
    'A_s': 2.1e-9,
    'n_s': 0.9684,
    'tau_reio': 0.056,
    'N_ur': 3.044,
    'T_cmb': T_FIRAS}

# Grid sizes.
N_from_centre_wb = 10
N_from_centre_wbc = 10
N_wb = 2*N_from_centre_wb + 1
N_wbc = 2*N_from_centre_wbc + 1

# Parameter ranges.
wb_min = 0.022
wb_max = 0.023
wbc_min = 0.14
wbc_max = 0.15

#Create grid values.
wb_values = np.linspace(wb_min, wb_max, N_wb)
wbc_values = np.linspace(wbc_min, wbc_max, N_wbc)

#Save parameter ranges.
with open("Parameter_ranges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"{wb_min:.10f}", f"{wb_max:.10f}", f"{N_wb}"])
    writer.writerow([f"{wbc_min:.10f}", f"{wbc_max:.10f}", f"{N_wbc}"])

#Prepare results array (2D: wb x wbc, with 2 derived parameters).
results = np.zeros((N_wb, N_wbc, 2))

for i, wb in enumerate(wb_values):
    print(f"\ri = {i+1} out of {N_wb}, please wait.....", end='', flush=True)
    for j, wbc in enumerate(wbc_values):
        wc = wbc - wb  # Dark matter density
        params = base_params.copy()
        params.update({'omega_b': wb, 'omega_cdm': wc})

        cosmo.set(params)
        cosmo.compute()
        derived = cosmo.get_current_derived_parameters(['z_rec', 'z_d'])
        results[i, j, 0] = derived['z_rec']
        results[i, j, 1] = derived['z_d']
    cosmo.struct_cleanup()
cosmo.empty()

#Convert z_star and z_d to photon temperatures at recombination and decoupling.
results = T_FIRAS*(1.0 + results)

# Create output directories if they don't exist
os.makedirs("T_star", exist_ok=True)
os.makedirs("T_d", exist_ok=True)

# Save results.
np.savetxt("T_star/T_star_values_FIRAS.csv", results[:, :, 0], fmt="%.10f", delimiter=",")
np.savetxt("T_d/T_d_values_FIRAS.csv",    results[:, :, 1], fmt="%.10f", delimiter=",")
print(f"\nResults saved.")
