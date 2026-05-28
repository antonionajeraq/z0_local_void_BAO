import numpy as np
from classy import Class
import sys
import os
import csv

cosmo = Class()
T_FIRAS = 2.72548 #Fixsen_2009.

base_params = {'output': '',
    'lensing': 'no',
    'A_s': 2.1e-9,
    'n_s': 0.9684,
    'tau_reio': 0.056,
    'N_ur': 3.044}
#N_eff from Jack J. Bennett et al. JCAP04 (2021), 073, DOI: 10.1088/1475-7516/2021/04/073
#A_s, n_s, and tau_reio from Arxiv:2506.20707. Neutrinos approximated as massless.

N_from_centre_wbt = 10
N_from_centre_wbct = 10
N_from_centre_z0 = 10

N_wbt = 2*N_from_centre_wbt + 1
N_wbct = 2*N_from_centre_wbct + 1
N_z0 = 2*N_from_centre_z0 + 1

wbt_min = 0.022 #Example grid
wbt_max = 0.023
wbct_min = 0.14
wbct_max = 0.15
z0_min = -0.01
z0_max = 0.03

wbt_values = np.linspace(wbt_min, wbt_max, N_wbt)
wbct_values = np.linspace(wbct_min, wbct_max, N_wbct)
z0_values = np.linspace(z0_min, z0_max, N_z0)

# Save parameter ranges.
with open("Parameter_ranges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"{wbt_min:.10f}", f"{wbt_max:.10f}", f"{N_wbt}"])
    writer.writerow([f"{wbct_min:.10f}", f"{wbct_max:.10f}", f"{N_wbct}"])
    writer.writerow([f"{z0_min:.10f}", f"{z0_max:.10f}", f"{N_z0}"])

results = np.zeros((N_wbt, N_wbct, N_z0, 2))
for i, wbt in enumerate(wbt_values):
    print(f"\ri = {i+1} out of {N_wbt}, please wait.....", end='', flush=True)

    for j, wbct in enumerate(wbct_values):
        for k, z0 in enumerate(z0_values):
            u = 1.0 + z0
            T_cmb = T_FIRAS*u

            v = u*u*u
            wb = wbt*v
            wbc = wbct*v
            wc = wbc - wb

            params = base_params.copy()
            params.update({'omega_b': wb,
                'omega_cdm': wc,
                'T_cmb': T_cmb})

            cosmo.set(params)
            cosmo.compute()

            derived = cosmo.get_current_derived_parameters(['z_rec','z_d'])

            results[i, j, k, 0] = derived['z_rec']
            results[i, j, k, 1] = derived['z_d']

    cosmo.struct_cleanup()

cosmo.empty()



#Convert z_star and z_d to photon temperatures at recombination and decoupling.
for k, z0 in enumerate(z0_values): results[:, :, k, :] = (T_FIRAS*(1.0 + z0))*(1.0 + results[:, :, k, :])



# Create output directories if they don't exist
os.makedirs("T_star", exist_ok=True)
os.makedirs("T_d", exist_ok=True)

for k, z0 in enumerate(z0_values):
    # Slice arrays: rows = wbt, columns = wbct
    Tstar_slice = results[:, :, k, 0]
    Td_slice    = results[:, :, k, 1]

    # Save without headers
    np.savetxt(f"T_star/T_star_values_z0_{k}.csv", Tstar_slice, fmt="%.10f", delimiter=",")
    np.savetxt(f"T_d/T_d_values_z0_{k}.csv", Td_slice, fmt="%.10f", delimiter=",")
