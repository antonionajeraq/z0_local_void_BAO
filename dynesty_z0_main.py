import numpy as np
import sys

#from dynesty_phen import H0_A_rd_mean

# We will set some flags to control what computations to perform
compute_minimum_chi2 = True # Compute the minimum chi squared contributions
perform_nested_sampling = True # Set to True to perform nested sampling
fix_z0 = False # Set to True to fix z0=0, False to let it vary
fixed_z0_value = 0.00 # Value to fix z0 to if fix_z0 is True
nested_sampling_data = "no_h0" # Options: "all", "CMB_only", "no_desi", "desi_only", "no_h0", "CMB+bbn+CC"
compute_Universe_age = False # This requires the nested sampling chains
compute_BAO_predictions = False # Compute BAO predictions for best-fit parameters for figure 4. Needs minimum chi2 first.
compute_BAO_significance_data_space = False # Compute BAO significance from CMB only chains
compute_BAO_significance_parameter_space = False # Compute BAO significance from parameter space volume reduction
compute_cosmological_parameters = False # Compute cosmological parameters for best-fit model
compute_z0_redshift_binning = True # Fit z0 at progressively lower redshift cutoffs (requires fix_z0=False mode)
compute_z0_H0_correlation = True # Compute weighted Pearson r(z0, H0) from an existing chain (requires fix_z0=False chain)

# --- Physical Constants ---
Sixth = 1 / 6
G = 6.6743e-11
c = 299792458
h_Planck = 6.62607e-34
hbar = h_Planck / (2 * np.pi)
k_Boltzmann = 1.380649e-23
eV = 1.602176634e-19
AU = 149597870700
Radian = 180 / np.pi  # Degrees
Degree = np.pi / 180
Arcsec = Degree / 3600
pc = AU / Arcsec
Mpc = pc * 1e6
H_100 = 100e3 / Mpc

GM_Sun = 1.32712440041279419e20
Year = 2*np.pi*np.sqrt(AU*AU*AU/GM_Sun)
Gyr = Year*1e9

# --- Cosmological Parameters ---
N_eff = 3.044  # effective neutrino number
Sum_mnu = 0.06 * eV / (c * c)
R_nu_gamma = N_eff * 7/8 * (4/11)**(4/3)
f_nu = R_nu_gamma / (1 + R_nu_gamma)

T_CMB_Firas = 2.72548  # K
rho_crit_100 = 3 * H_100**2 / (8 * np.pi * G)

T_d_ref_CMB = 2892.7189265390 # Reference value for the temperature of the CMB at drag epoch, in K.
T_d_ref = np.array([0.02250, 0.1450, 0.01]) # Reference values for the temperature at drag computation.
T_d_gradient = np.array([5926.7982685835, 203.2921847528, -2.3736921657]) # Gradient for the temperature at drag computation.
T_d_hessian = np.array([[-189596.6848072170, -4765.7156601812, 108.8993400754], 
                        [-4765.7156601812, -624.6877708911, -5.3358430973],
                        [108.8993400754, -5.3358430973, 17.1094164296]]) # Hessian for the temperature at drag computation.

T_star_ref_CMB = 2970.2490798866 # Reference value for the temperature of the CMB at recombination, in K.
T_star_ref = np.array([0.02250, 0.1450, 0.01]) # Reference values for the temperature at recombination computation.
T_star_gradient = np.array([-3187.3522386402, 194.0022355988, -3.8997457516]) # Gradient for the temperature at recombination computation.
T_star_hessian = np.array([[265184.9625445241, -7854.3444118611, 144.6125754140],
                           [-7854.3444118611, -491.8071934238, -9.3247907229],
                           [144.6125754140, -9.3247907229, 23.4632437267]]) # Hessian for the temperature at recombination computation.

def functions_of_Tcmb(T_CMB):
    rho_gamma = np.pi**2 / 15 * (k_Boltzmann**4) / (hbar**3 * c**5) * T_CMB**4

    # --- Photon and Neutrino Densities ---
    zeta_3 = 1.202056903159594  # ζ(3)
    n_gamma = 16 * np.pi * zeta_3 * ((k_Boltzmann * T_CMB / (h_Planck * c))**3)
    n_nu = 3/11 * n_gamma * (N_eff / 3)**0.68
    rho_nu = n_nu * Sum_mnu
    rho_nu_ur = rho_gamma * R_nu_gamma
    a_nr_sqinv = (rho_nu**2) / (rho_nu_ur**2) - 1
    a_nr_sq = 1 / a_nr_sqinv

    w_nu = rho_nu / rho_crit_100
    #print(f"w_nu = {w_nu:.7e}")
    w_gamma = rho_gamma / rho_crit_100
    w_nu_ur = rho_nu_ur / rho_crit_100
    return w_gamma, w_nu, w_nu_ur, a_nr_sq, rho_gamma, rho_nu, rho_nu_ur

import numpy as np

# --- DESI DR2 BAO Data (all 7 bins including BGS at z=0.295) ---
z_DESI = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
Data_DESI = np.array([7.942, 13.588, 21.863, 17.351, 19.455, 21.576, 17.641, 27.601, 14.176, 30.512, 12.817, 38.988, 8.632])
C_blocks_DESI = [0.075**2, np.array([[0.167**2, -0.459*0.167*0.425], [-0.459*0.167*0.425, 0.425**2]]),
             np.array([[0.177**2, -0.404*0.177*0.330], [-0.404*0.177*0.330, 0.330**2]]),
             np.array([[0.152**2, -0.416*0.152*0.193], [-0.416*0.152*0.193, 0.193**2]]),
             np.array([[0.318**2, -0.434*0.318*0.221], [-0.434*0.318*0.221, 0.221**2]]),
             np.array([[0.760**2, -0.500*0.760*0.516], [-0.500*0.760*0.516, 0.516**2]]),
             np.array([[0.531**2, -0.431*0.531*0.101], [-0.431*0.531*0.101, 0.101**2]])]

# 13x13 covariance matrix: first element is DV/rd (BGS), rest are DM/rd, DH/rd pairs
Cov_DESI = np.zeros((13, 13))
Cov_DESI[0, 0] = C_blocks_DESI[0]
for i, Ci in enumerate(C_blocks_DESI[1:]):
    row = 2*i + 1
    Cov_DESI[row:row+2, row:row+2] = Ci

C_inv_total = np.linalg.inv(Cov_DESI)
C_inv_total = np.array(C_inv_total)

z_CC, H_z_CC, sigma_H_z_CC = np.loadtxt('CC_data.txt', unpack=True)

print(f"Cosmic Chronometers data points used: {len(z_CC)}")

from scipy.special import lambertw
from scipy.optimize import root_scalar

def reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq):
    Omega_Lambda = 1 - Omega_bc - Omega_gamma - Omega_nu
    E = np.sqrt(Omega_bc * (1 + z)**3 + Omega_gamma*(1+z)**4 + Omega_Lambda + Omega_nu_ur * np.sqrt(1+1/((1+z)**2*a_nr_sq)) * (1+z)**4 )
    return E

def integrand_rec(u, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0):
    z = np.sinh(u)
    E = reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq)
    return np.cosh(u) / E

def integrand_age_rec(u, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0):
    z = np.sinh(u)
    E = reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq)
    return np.cosh(u) / (E*(1 + z))

def integrand(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0):
    E = reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq)
    return 1 / E

def integrand_Einv_simple(z, Omega_M):
    E = np.sqrt(Omega_M * (1 + z)**3 + (1 - Omega_M))
    return 1 / E

from scipy.integrate import quad

def comoving_distance_arcsinh(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, z_star):
    integral, _ = quad(integrand_rec, 0, np.arcsinh(z_star), args=(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0))
    return c / H0 * integral  # Return in metres.

def comoving_distance(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, z):
    integral, _ = quad(integrand, 0, z, args=(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0))
    return c / H0 * integral  # Return in metres.

comoving_distance_vec = np.vectorize(comoving_distance, excluded=['Omega_bc', 'Omega_gamma', 'Omega_nu', 'Omega_nu_ur', 'a_nr_sq', 'H0'])

def model_predictions(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, r_d, z0, extended_z=False):
    z_cosmo = (z - z0)/(1 + z0)
    DM_array = comoving_distance_vec(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq, H0, z_cosmo)
    DH_array = c / (H0 * (1 + z0) * reduced_hubble_factor(z_cosmo, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_sq))  # DH in Mpc
    if extended_z:
        DV = np.power((DM_array[0] ** 2) * DH_array[0] * z[0], 1.0 / 3.0)
        interleaved = np.stack([DM_array[1:], DH_array[1:]], axis=1).reshape(-1)
        vec = np.concatenate([np.array([DV]), interleaved])
    else:
        vec = np.stack([DM_array[0:], DH_array[0:]], axis=1).reshape(-1)
    #vec = np.concatenate([np.array([DV]), interleaved]) / r_d
    return vec/r_d#vec

mu_cmb = np.array([1.04161312, 0.02238286, 0.14247121]) # Canphuis et al. 2025

cov_cmb = 1.0e-9 * np.array([[ 54.39601311,   0.80698092, -20.64886707],
                            [  0.80698092,   8.54260314, -13.26477391],
                            [-20.64886707, -13.26477391, 692.15996597]])

inv_cov_cmb = np.linalg.inv(cov_cmb)

def theta_star(w_b, w_bc, h, z0, print_z_rec = False):
    w_b_tilde = w_b/(1 + z0)**3
    w_bc_tilde = w_bc/(1 + z0)**3
    rho_b = w_b * rho_crit_100
    rho_bc = w_bc * rho_crit_100

    T_cmb_mod = T_CMB_Firas * (1 + z0)

    w_gamma, w_nu, w_nu_ur, a_nr_sq, rho_gamma, rho_nu, rho_nu_ur = functions_of_Tcmb(T_cmb_mod)

    d_drag = np.array([w_b_tilde, w_bc_tilde, z0]) - T_d_ref  # Use the parameters to compute d_drag
    T_drag = T_d_ref_CMB + T_d_gradient @ d_drag + 0.5 * d_drag @ T_d_hessian @ d_drag
    z_drag = T_drag / T_cmb_mod - 1
    a_drag = 1 / (1 + z_drag)

    d_rec = np.array([w_b_tilde, w_bc_tilde, z0]) - T_star_ref  # Use the parameters to compute d_rec
    T_rec = T_star_ref_CMB + T_star_gradient @ d_rec + 0.5 * d_rec @ T_star_hessian @ d_rec
    z_rec = T_rec / T_cmb_mod - 1
    if print_z_rec:
        print(f"Computed z_rec: {z_rec:.3f}")
    a_rec = 1 / (1 + z_rec)

    R_b_gamma = rho_b / rho_gamma
    R_bc_gamma = rho_bc / rho_gamma
    R_0 = 3/4 * R_b_gamma  # Multiply by a for other epochs

    a_eq = (1 + R_nu_gamma) / R_bc_gamma
    z_eq = 1 / a_eq - 1

    # --- Ratios ---
    R_eq = R_0 * a_eq
    R_rec = R_0 * a_rec

    # --- Sound Horizon ---
    u = (np.sqrt(1 + R_rec) + np.sqrt(R_rec + R_eq)) / (1 + np.sqrt(R_eq))
    r_rec = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u)

    DM_rec = comoving_distance_arcsinh(w_bc / h**2, w_gamma / h**2, w_nu / h**2, w_nu_ur / h**2, a_nr_sq, h * H_100, z_rec)

    theta_star_value = r_rec / DM_rec

    return theta_star_value

#theta_star(0.021361, 0.261947*0.72014951**2, 0.72014951, -0.015475, print_z_rec=True)
#sys.exit()

def chi2_cmb(w_bc, w_b, H0, z0):
    
    theta_star_value = 100*theta_star(w_b, w_bc, H0/100, z0)

    mu_model = np.array([theta_star_value, w_b/(1 + z0)**3, w_bc/(1 + z0)**3])

    chi2_cmb = (mu_model - mu_cmb).T @ inv_cov_cmb @ (mu_model - mu_cmb)

    return chi2_cmb  # Return log likelihood

from scipy.stats import multivariate_normal, norm

def log_prior(theta):
    w_b, omega_m, H0, z0 = theta
    if not (0.00 < omega_m < 0.50):
        return -np.inf
    if not (50.0 < H0 < 100.0):
        return -np.inf
    if not (0.020 < w_b < 0.025):
        return -np.inf
    if not (-0.1 < z0 < 0.1):
        return -np.inf
    return 0  # uniform priors for omega_m and H0, Gaussian for r_d

H0_SHOES = 73.17  # Hubble constant from SH0ES in km/s/Mpc
eH0_SHOES = 0.86  # Uncertainty in H0_SHOES in km/s/Mpc
H0_SNII = 74.9  # Hubble constant from SNII in km/s/Mpc
eH0_SNII = 2.7  # Uncertainty in H0_SNII
H0_SBF = 73.8  # Hubble constant from SBF in km/s/Mpc
eH0_SBF = 2.4  # Uncertainty in H0_SBF
H0_maser = 73.9  # Hubble constant from maser in km/s/Mpc
eH0_maser = 3.0  # Uncertainty in H0_maser

w_b_bbn = 0.02241 # APSS,371,2; 2511.06275
ew_b_bbn = 0.00031

def log_likelihood(theta, use_desi=True, use_h0=True, use_cc=True, use_cmb=True, use_bbn=True, desi_only=False, print_r_d = False):
    if not desi_only:
        if fix_z0:
            z0 = fixed_z0_value
            w_b, omega_m, H0 = theta
        else: w_b, omega_m, H0, z0 = theta

        h = H0 / 100.0
    
        w_bc = omega_m * h**2
        w_b_tilde = w_b/(1 + z0)**3
        w_bc_tilde = w_bc/(1 + z0)**3

    else: 
        if fix_z0:
            z0 = fixed_z0_value
            omega_m, H0_rd = theta
        else: omega_m, H0_rd, z0 = theta
        
    chi2_total = 0
    log_det_total = 0
    
    T_cmb_mod = T_CMB_Firas * (1 + z0)
    w_gamma, w_nu, w_nu_ur, a_nr_sq, rho_gamma, rho_nu, rho_nu_ur = functions_of_Tcmb(T_cmb_mod)

    # CMB term (always computed if use_cmb=True)
    if use_cmb:
        chi2_CMB = chi2_cmb(w_bc, w_b, H0, z0)
        chi2_total += chi2_CMB
        log_det_total += np.log(2 * np.pi * np.linalg.det(cov_cmb))

    if use_bbn:
        chi2_BBN = ((w_b_tilde - w_b_bbn) / ew_b_bbn)**2
        chi2_total += chi2_BBN
        log_det_total += np.log(2 * np.pi * ew_b_bbn**2)
    
    # DESI term
    if use_desi:
        if not desi_only:
            d_drag = np.array([w_b_tilde, w_bc_tilde, z0]) - T_d_ref  # Use the parameters to compute d_drag
            T_drag = T_d_ref_CMB + T_d_gradient @ d_drag + 0.5 * d_drag @ T_d_hessian @ d_drag
            z_drag = T_drag / T_cmb_mod - 1
            a_drag = 1 / (1 + z_drag)

            rho_b = w_b * rho_crit_100
            rho_bc = w_bc * rho_crit_100

            R_b_gamma = rho_b / rho_gamma
            R_bc_gamma = rho_bc / rho_gamma
            R_0 = 3/4 * R_b_gamma  # Multiply by a for other epochs

            a_eq = (1 + R_nu_gamma) / R_bc_gamma
            z_eq = 1 / a_eq - 1
            # --- Ratios ---
            R_eq = R_0 * a_eq
            R_drag = R_0 * a_drag

            # --- Sound Horizon ---
            u = (np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_eq)) / (1 + np.sqrt(R_eq))
            r_d = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u)
            if print_r_d:
                print(f"r_d = {r_d/Mpc:.3f} Mpc")
            predictions = model_predictions(z_DESI, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, h*H_100, r_d, z0, True)
        else:
            predictions = model_predictions(z_DESI, omega_m, 0, 0, 0, a_nr_sq, 1000*H0_rd, 1, z0, True)
        diff = predictions - Data_DESI
        chi2_total += diff @ C_inv_total @ diff
        log_det_total += np.log(2 * np.pi * np.linalg.det(Cov_DESI))
    
    # H0 measurements
    if use_h0:
        chi2_total += ((H0 - H0_SHOES) / eH0_SHOES)**2
        chi2_total += ((H0 - H0_SNII) / eH0_SNII)**2
        chi2_total += ((H0 - H0_SBF) / eH0_SBF)**2
        chi2_total += ((H0 - H0_maser) / eH0_maser)**2
        log_det_total += np.log(2 * np.pi * eH0_SHOES**2)
        log_det_total += np.log(2 * np.pi * eH0_SNII**2)
        log_det_total += np.log(2 * np.pi * eH0_SBF**2)
        log_det_total += np.log(2 * np.pi * eH0_maser**2)
    
    # Cosmic Chronometers
    if use_cc:
        z_CC_cosmo = (z_CC - z0) / (1 + z0)
        H_z_predicted = H0 * (1 + z0) * reduced_hubble_factor(z_CC_cosmo, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq)
        chi2_total += np.sum(((H_z_predicted - H_z_CC) / sigma_H_z_CC)**2)
        log_det_total += np.sum(np.log(2 * np.pi * sigma_H_z_CC**2))

    return -0.5 * (chi2_total + log_det_total)

def chi_squared(theta, use_desi=True, use_h0=False, use_cc=True, use_cmb=True, use_bbn = True, desi_only=False, print_output=False):
    if not desi_only:
        if fix_z0:
            z0 = fixed_z0_value
            w_b, omega_m, H0 = theta
        else: w_b, omega_m, H0, z0 = theta

        h = H0 / 100.0
    
        w_bc = omega_m * h**2
    else: 
        if fix_z0:
            z0 = fixed_z0_value
            omega_m, H0_rd = theta
        else: omega_m, H0_rd, z0 = theta
    
    chi2_total = 0

    w_b_tilde = w_b/(1 + z0)**3
    w_bc_tilde = w_bc/(1 + z0)**3
    T_cmb_mod = T_CMB_Firas * (1 + z0)
    w_gamma, w_nu, w_nu_ur, a_nr_sq, rho_gamma, rho_nu, rho_nu_ur = functions_of_Tcmb(T_cmb_mod)
    
    # Additional safety check: ensure w_b_tilde and w_bc_tilde are positive
    #if w_b_tilde <= 0 or w_bc_tilde <= 0 or not np.isfinite(w_b_tilde) or not np.isfinite(w_bc_tilde):
    #    return -np.inf
    
    # CMB term (always computed if use_cmb=True)
    if use_cmb:
        chi2_CMB = chi2_cmb(w_bc, w_b, H0, z0)
        chi2_total += chi2_CMB
    
    if use_bbn:
        chi2_BBN = ((w_b_tilde - w_b_bbn) / ew_b_bbn)**2
        chi2_total += chi2_BBN
    
    # DESI term
    if use_desi:
        if not desi_only:
            d_drag = np.array([w_b_tilde, w_bc_tilde, z0]) - T_d_ref  # Use the parameters to compute d_drag
            T_drag = T_d_ref_CMB + T_d_gradient @ d_drag + 0.5 * d_drag @ T_d_hessian @ d_drag
            z_drag = T_drag / T_cmb_mod - 1
            a_drag = 1 / (1 + z_drag)

            rho_b = w_b * rho_crit_100
            rho_bc = w_bc * rho_crit_100

            R_b_gamma = rho_b / rho_gamma
            R_bc_gamma = rho_bc / rho_gamma
            R_0 = 3/4 * R_b_gamma  # Multiply by a for other epochs

            a_eq = (1 + R_nu_gamma) / R_bc_gamma
            z_eq = 1 / a_eq - 1
            # --- Ratios ---
            R_eq = R_0 * a_eq
            R_drag = R_0 * a_drag

            # --- Sound Horizon ---
            u = (np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_eq)) / (1 + np.sqrt(R_eq))
            r_d = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u)

            predictions = model_predictions(z_DESI, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, h*H_100, r_d, z0, True)
        else:
            predictions = model_predictions(z_DESI, omega_m, 0, 0, 0, a_nr_sq, 1000*H0_rd, 1, z0, True)

        diff = predictions - Data_DESI
        chi2_total += diff @ C_inv_total @ diff
    # H0 measurements
    if use_h0:
        chi2_total += ((H0 - H0_SHOES) / eH0_SHOES)**2
        chi2_total += ((H0 - H0_SNII) / eH0_SNII)**2
        chi2_total += ((H0 - H0_SBF) / eH0_SBF)**2
        chi2_total += ((H0 - H0_maser) / eH0_maser)**2
    
    # Cosmic Chronometers
    if use_cc:
        z_CC_cosmo = (z_CC - z0) / (1 + z0)
        H_z_predicted = H0 * (1 + z0) * reduced_hubble_factor(z_CC_cosmo, omega_m, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq)
        chi2_total += np.sum(((H_z_predicted - H_z_CC) / sigma_H_z_CC)**2)

    if print_output:
        print(f"Chi2 contributions:")
        if use_cmb:
            print(f"  CMB: {chi2_CMB:.3f}")
        if use_bbn:
            print(f"  BBN: {chi2_BBN:.3f}")
        if use_desi:
            print(f"  DESI: {diff @ C_inv_total @ diff:.3f}")
        if use_h0:
            chi2_h0 = ((H0 - H0_SHOES) / eH0_SHOES)**2 + ((H0 - H0_SNII) / eH0_SNII)**2 + ((H0 - H0_SBF) / eH0_SBF)**2 + ((H0 - H0_maser) / eH0_maser)**2
            print(f"  H0: {chi2_h0:.3f}")
        if use_cc:
            chi2_cc = np.sum(((H_z_predicted - H_z_CC) / sigma_H_z_CC)**2)
            print(f"  CC: {chi2_cc:.3f}")
        print(f"Total Chi2: {chi2_total:.3f}")

    # Final check for NaN or inf values
    if not np.isfinite(chi2_total):
        return -np.inf
    
    return chi2_total

#theta_test = [0.022381, 0.3166, 67.05, 0.05]
#print(f"Test chi-squared at theta={theta_test}: {chi_squared(theta_test, print_output=True):.3f}")
#sys.exit()

def minimize_chi2(use_desi=True, use_h0=False, use_cc=True, use_cmb=True, desi_only=False):
    from scipy.optimize import minimize

    if not desi_only:
        if fix_z0:
            z0 = fixed_z0_value
            initial_guess = [0.022, 0.3, 70.0]
        else:
            initial_guess = [0.022, 0.3, 70.0, 0.0]
    else:
        if fix_z0:
            z0 = fixed_z0_value
            initial_guess = [0.3, 7500.0]
        else:
            initial_guess = [0.3, 7500.0, 0.0]
    
    # Create chi_squared function with chosen flags
    def chi2_func(theta):
        return chi_squared(theta, use_desi=use_desi, use_h0=use_h0, 
                          use_cc=use_cc, use_cmb=use_cmb, desi_only=desi_only)
    
    # Use bounds to constrain the search
    if not desi_only:
        if fix_z0:
            bounds = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0)]
        else:
            bounds = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0), (-0.09, 0.09)]
    else:
        if fix_z0:
            bounds = [(0.01, 0.49), (5000, 13000)]
        else:
            bounds = [(0.01, 0.49), (5000, 13000), (-0.09, 0.09)]
    
    result = minimize(chi2_func, initial_guess, method='L-BFGS-B', bounds=bounds)

    if result.success:
        fitted_params = result.x
        print("Fitted parameters:")
        if not desi_only:
            if fix_z0:
                print(f"w_b: {fitted_params[0]:.6f}, omega_m: {fitted_params[1]:.6f}, H0: {fitted_params[2]:.6f}, z0: {fixed_z0_value:.6f}")
            else:
                print(f"w_b: {fitted_params[0]:.6f}, omega_m: {fitted_params[1]:.6f}, H0: {fitted_params[2]:.6f}, z0: {fitted_params[3]:.6f}")
        else:
            if fix_z0:
                print(f"omega_m: {fitted_params[0]:.6f}, H0_rd: {fitted_params[1]:.6f}, z0: {fixed_z0_value:.6f}")
            else:
                print(f"omega_m: {fitted_params[0]:.6f}, H0_rd: {fitted_params[1]:.6f}, z0: {fitted_params[2]:.6f}")
        chi2_min = chi_squared(fitted_params, use_desi=use_desi, use_h0=use_h0, 
                              use_cc=use_cc, use_cmb=use_cmb, desi_only=desi_only, 
                              print_output=True)
    else:
        print("Optimization failed.")
    return result

def minimize_chi2_sequential(use_desi=True, use_h0=False, use_cc=True, use_cmb=True, desi_only=False):
    """
    Minimize chi-squared in two steps:
    1. First with z0 fixed to 0
    2. Then with z0 free, using the first result as starting point
    """
    from scipy.optimize import minimize

    print("=" * 70)
    print("STEP 1: Minimizing with z0 FIXED to 0")
    print("=" * 70)
    
    # Create chi_squared function with z0=0 fixed
    if not desi_only:
        def chi2_fixed_z0(theta_reduced):
            w_b, omega_m, H0 = theta_reduced
            theta_full = [w_b, omega_m, H0, fixed_z0_value]
            return chi_squared(theta_full, use_desi=use_desi, use_h0=use_h0,
                              use_cc=use_cc, use_cmb=use_cmb, desi_only=desi_only)

        initial_guess_fixed = [0.022, 0.3166, 67.2]
        bounds_fixed = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0)]
    else:
        def chi2_fixed_z0(theta_reduced):
            omega_m, H0_rd = theta_reduced
            theta_full = [omega_m, H0_rd, fixed_z0_value]
            return chi_squared(theta_full, use_desi=use_desi, use_h0=use_h0,
                              use_cc=use_cc, use_cmb=use_cmb, desi_only=desi_only)
        
        initial_guess_fixed = [0.3, 7500.0]
        bounds_fixed = [(0.01, 0.49), (5000, 13000)]
    
    result_fixed = minimize(chi2_fixed_z0, initial_guess_fixed, method='L-BFGS-B', bounds=bounds_fixed)
    
    if result_fixed.success:
        fitted_params_fixed = result_fixed.x
        print("\nFitted parameters (z0=0 FIXED):")
        if not desi_only:
            print(f"w_b: {fitted_params_fixed[0]:.6f}, omega_m: {fitted_params_fixed[1]:.6f}, H0: {fitted_params_fixed[2]:.6f}, z0: {fixed_z0_value:.6f}")
            theta_full_fixed = [*fitted_params_fixed, fixed_z0_value]
        else:
            print(f"omega_m: {fitted_params_fixed[0]:.6f}, H0_rd: {fitted_params_fixed[1]:.6f}, z0: {fixed_z0_value:.6f}")
            theta_full_fixed = [*fitted_params_fixed, fixed_z0_value]
        
        chi2_fixed = chi_squared(theta_full_fixed, use_desi=use_desi, use_h0=use_h0,
                                use_cc=use_cc, use_cmb=use_cmb, desi_only=desi_only,
                                print_output=True)
        T_cmb_mod = T_CMB_Firas
        w_gamma, w_nu, w_nu_ur, a_nr_sq, rho_gamma, rho_nu, rho_nu_ur = functions_of_Tcmb(T_cmb_mod)
        predicted_bao = model_predictions(z_DESI, fitted_params_fixed[1], w_gamma/(fitted_params_fixed[2]/100)**2, w_nu/(fitted_params_fixed[2]/100)**2, w_nu_ur/(fitted_params_fixed[2]/100)**2, a_nr_sq, (fitted_params_fixed[2]/100)*H_100, 147.05 * Mpc * (fitted_params_fixed[0]/0.02236)**-0.13 * (fitted_params_fixed[1]/0.1432)**-0.23 * (N_eff/3.04)**-0.1, fix_z0, True)
        #print("\nPredicted DESI BAO values (z0=0 fixed):")
        #print(predicted_bao)
    else:
        print("Optimization with z0=0 failed.")
        return None, None
    
    print("\n" + "=" * 70)
    print("STEP 2: Minimizing with z0 FREE")
    print("=" * 70)
    
    # Now minimize with z0 free, starting from previous result
    if not desi_only:
        initial_guess_free = [*fitted_params_fixed, fixed_z0_value]
        bounds_free = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0), (-0.1, 0.1)]
    else:
        initial_guess_free = [*fitted_params_fixed, fixed_z0_value]
        bounds_free = [(0.01, 0.49), (5000, 13000), (-0.1, 0.1)]
    
    def chi2_free_z0(theta):
        return chi_squared(theta, use_desi=use_desi, use_h0=use_h0,
                          use_cc=use_cc, use_cmb=use_cmb, desi_only=desi_only)
    
    result_free = minimize(chi2_free_z0, initial_guess_free, method='L-BFGS-B', bounds=bounds_free)
    
    if result_free.success:
        fitted_params_free = result_free.x
        print("\nFitted parameters (z0 FREE):")
        if not desi_only:
            print(f"w_b: {fitted_params_free[0]:.6f}, omega_m: {fitted_params_free[1]:.6f}, H0: {fitted_params_free[2]:.6f}, z0: {fitted_params_free[3]:.6f}")
        else:
            print(f"omega_m: {fitted_params_free[0]:.6f}, H0_rd: {fitted_params_free[1]:.6f}, z0: {fitted_params_free[2]:.6f}")
        
        chi2_free = chi_squared(fitted_params_free, use_desi=use_desi, use_h0=use_h0,
                               use_cc=use_cc, use_cmb=use_cmb, desi_only=desi_only,
                               print_output=True)

        T_cmb_mod = T_CMB_Firas * (1 + fitted_params_free[3])
        w_gamma, w_nu, w_nu_ur, a_nr_sq, rho_gamma, rho_nu, rho_nu_ur = functions_of_Tcmb(T_cmb_mod)
        predicted_bao = model_predictions(z_DESI, fitted_params_free[1], w_gamma/(fitted_params_free[2]/100)**2, w_nu/(fitted_params_free[2]/100)**2, w_nu_ur/(fitted_params_free[2]/100)**2, a_nr_sq, (fitted_params_free[2]/100)*H_100, 147.05 * Mpc * (fitted_params_free[0]/0.02236)**-0.13 * (fitted_params_free[1]/0.1432)**-0.23 * (N_eff/3.04)**-0.1, fixed_z0_value, True)
        #print("\nPredicted DESI BAO values (z0=0 fixed):")
        # Save minimum chi2 predicted BAO values for comparison
        #with open("predicted_bao_z0_free.txt", "w") as f:
        #    for z_val, bao_val in zip(z_DESI, predicted_bao): 
        #        f.write(f"{z_val:.3f} {bao_val:.6f}\n") 
        #print(predicted_bao)
        
        print("\n" + "=" * 70)
        print("COMPARISON:")
        print("=" * 70)
        print(f"Chi2 (z0=0 fixed):  {chi2_fixed:.3f}")
        print(f"Chi2 (z0 free):     {chi2_free:.3f}")
        print(f"Delta Chi2:         {(chi2_fixed - chi2_free):.3f}")
        print(f"Significance:       {np.sqrt((chi2_fixed - chi2_free)):.2f} sigma (if Delta Chi2 > 0)")
        print("=" * 70)
    else:
        print("Optimization with z0 free failed.")
        return result_fixed, None
    
    return result_fixed, result_free

# Then call with flags:
# result = minimize_chi2(use_h0=False, use_desi=True)

if compute_minimum_chi2:
    # Option 1: Standard minimization with flags
    #results = minimize_chi2(use_h0=False, use_desi=True, use_cc=True, use_cmb=True)

    # Option 2: Sequential minimization (z0 fixed then free)  
    results_fixed, results_free = minimize_chi2_sequential(use_desi=True, use_h0=False, use_cc=True, use_cmb=True)

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

import numpy as np
from dynesty import NestedSampler
from scipy.stats import norm

# --- Define the prior transform ---
def prior_transform(u):
    if not nested_sampling_data == "desi_only":
        # u: array of numbers in [0, 1]
        w_b = 0.020 + 0.005 * u[0]        # [0.020, 0.025]
        omega_m = 0.01 + 0.48 * u[1]         # [0.01, 0.49]
        H0 = 50.0 + 50.0 * u[2]              # [50.0, 100.0]
        if fix_z0:
            return [w_b, omega_m, H0]  # Only 3 parameters when z0 is fixed
        else:
            z0 = -0.05 + 0.1 * u[3] # [-0.1, 0.1]
            return [w_b, omega_m, H0, z0]
    else:
        omega_m = 0.01 + 0.48 * u[0]         # [0.01, 0.49]
        H0_rd = 5000 + 8000 * u[1]         # [0.050, 0.100] in units of 1000 km/s/Mpc
        if fix_z0:
            return [omega_m, H0_rd]  # Only 2 parameters when z0 is fixed
        else:
            z0 = -0.05 + 0.1 * u[2] # [-0.1, 0.1]
            return [omega_m, H0_rd, z0]

# --- Set up and run dynesty ---
if not nested_sampling_data == "desi_only":
    ndim = 3 if fix_z0 else 4
else:
    ndim = 2 if fix_z0 else 3
nprocs = 4  # or however many CPU cores you want to use

import multiprocessing

if perform_nested_sampling:
    if not fix_z0:
        print("Performing nested sampling with z0 free...")
    # Define which datasets to use
    if nested_sampling_data == "CMB_only":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=False, use_h0=False, use_cc=False, use_cmb=True, use_bbn=False)
    elif nested_sampling_data == "all":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=True, use_h0=True, use_cc=True, use_cmb=True, use_bbn=True)
    elif nested_sampling_data == "no_desi":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=False, use_h0=True, use_cc=True, use_cmb=True, use_bbn=True)
    elif nested_sampling_data == "desi_only":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=True, use_h0=False, use_cc=False, use_cmb=False, use_bbn=False, desi_only=True)
    elif nested_sampling_data == "no_h0":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=True, use_h0=False, use_cc=True, use_cmb=True, use_bbn=True)
    elif nested_sampling_data == "CMB+bbn+CC":
        def likelihood_func(theta):
            return log_likelihood(theta, use_desi=False, use_h0=False, use_cc=True, use_cmb=True, use_bbn=True)
    else:
        likelihood_func = log_likelihood
    
    with multiprocessing.Pool(processes=nprocs) as pool:
        sampler = NestedSampler(
            likelihood_func,
            prior_transform,
            ndim,
            nlive=500,
            bound='multi',
            sample='rwalk',
            pool=pool,
            queue_size=nprocs
        )
        sampler.run_nested(dlogz=0.01, print_progress=True)
        results = sampler.results

    # --- Extract evidence and samples ---
    logZ = results.logz[-1]
    logZerr = results.logzerr[-1]
    print(f"Log evidence: {logZ:.3f} ± {logZerr:.3f}")

    # Posterior samples (weighted)
    from dynesty import utils as dyfunc
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    mean = np.average(samples, axis=0, weights=weights)
    print("Posterior mean:", mean)

    #if not nested_sampling_data == "desi_only":
        # If your log_posterior can accept a 2D array, you can do:
    #    log_posteriors = np.apply_along_axis(log_posterior, 1, samples)

    #    print("Log posteriors shape:", log_posteriors.shape)
    if nested_sampling_data == "CMB_only":
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_CMB_only_ext_z.npz"
            else:
                filename = f"dynesty_results_z0_CMB_only_ext_{fixed_z0_value}.npz"
            np.savez(filename, samples=samples, weights=weights)
        else:
            filename = f"dynesty_results_z0_CMB_only_ext.npz"
            np.savez(filename, samples=samples, weights=weights)
    elif nested_sampling_data == "no_desi":
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_no_desi_ext_z.npz"
            else:
                filename = f"dynesty_results_z0_no_desi_ext_{fixed_z0_value}.npz"
            np.savez(filename, samples=samples, weights=weights)
        else:
            filename = f"dynesty_results_z0_no_desi_ext.npz"
            np.savez(filename, samples=samples, weights=weights)
    elif nested_sampling_data == "desi_only":
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_desi_only_ext_z.npz"
            else:
                filename = f"dynesty_results_z0_desi_only_ext_{fixed_z0_value}.npz"
            np.savez(filename, samples=samples, weights=weights)
        else:
            filename = f"dynesty_results_z0_desi_only_ext.npz"
            np.savez(filename, samples=samples, weights=weights)
    elif nested_sampling_data == "no_h0":
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_no_h0_ext_z.npz"
            else:
                filename = f"dynesty_results_z0_no_h0_ext_{fixed_z0_value}.npz"
            np.savez(filename, samples=samples, weights=weights)
        else:
            filename = f"dynesty_results_z0_no_h0_ext.npz"
            np.savez(filename, samples=samples, weights=weights)
    elif nested_sampling_data == "CMB+bbn+CC":
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_CMB+bbn+CC_ext_z.npz"
            else:
                filename = f"dynesty_results_z0_CMB+bbn+CC_ext_{fixed_z0_value}.npz"
            np.savez(filename, samples=samples, weights=weights)
        else:
            filename = f"dynesty_results_z0_CMB+bbn+CC_ext.npz"
            np.savez(filename, samples=samples, weights=weights)
    else:
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_ext_z.npz"
            else:
                filename = f"dynesty_results_z0_ext_{fixed_z0_value}.npz"
            np.savez(filename, samples=samples, weights=weights)
        else:
            filename = f"dynesty_results_z0_ext.npz"
            np.savez(filename, samples=samples, weights=weights)

def age_of_universe(Omega_bc, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0):
    h = H0 / 100.0
    def integrand(a):
        z = 1 / a - 1
        E = a * reduced_hubble_factor(z, Omega_bc, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq)
        return 1.0 / E

    integral, _ = quad(integrand, 0, 1)
    t_0 = integral / H0 # H0 in km/s/Mpc
    t_0 = t_0 * Mpc / (1e3)  # Convert to seconds
    t_0 = t_0 / Gyr  # Convert seconds to Gyr
    return t_0

#def reduced_hubble_factor_cpl(z, Omega_bc, w_gamma, w_nu, w_nu_ur, a_nr_sq, w0, wa):
#    Omega_Lambda = 1 - Omega_bc - w_gamma - w_nu
#    w_DE = w0 + wa * (z / (1 + z))
#    E = np.sqrt(Omega_bc * (1 + z)**3 + w_gamma*(1+z)**4 + Omega_Lambda * (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z)) + w_nu_ur * np.sqrt(1+1/((1+z)**2*a_nr_sq)) * (1+z)**4 )
#    return E

#def age_of_universe_cpl(Omega_bc, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0, w0, wa):
#    h = H0 / 100.0
#    def integrand(a):
#        z = 1 / a - 1
#        E = a * reduced_hubble_factor_cpl(z, Omega_bc, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, w0, wa)
#        return 1.0 / E

#    integral, _ = quad(integrand, 0, 1)
#    t_0 = integral / H0 # H0 in km/s/Mpc
#    t_0 = t_0 * Mpc / (1e3)  # Convert to seconds
#    t_0 = t_0 / Gyr  # Convert seconds to Gyr
#    return t_0

#Omega_bc_cpl = 0.353
#H0_cpl = 63.6
#w0_cpl = -0.42
#wa_cpl = -1.75

if compute_Universe_age:
    try:
        data = np.load("dynesty_results_LCDM_final.npz")
        samples = data['samples']
        weights = data['weights']
    except FileNotFoundError:
        raise FileNotFoundError("Nested sampling results file not found. Please run nested sampling first.")

    ages = np.array([age_of_universe(theta[1], w_gamma, w_nu, w_nu_ur, a_nr_sq, theta[2]) for theta in samples])
    mean_age = np.average(ages, weights=weights)
    variance_age = np.average((ages - mean_age)**2, weights=weights)
    std_age = np.sqrt(variance_age)

    print(f"Universe Age from Nested sampling chain: {mean_age:.3f} ± {std_age:.3f} Gyr")

    #print("Universe Age for CPL best-fit parameters:")
    #age_cpl = age_of_universe_cpl(Omega_bc_cpl, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0_cpl, w0_cpl, wa_cpl)
    #print(f"Age (CPL best-fit): {age_cpl:.3f} Gyr")

if compute_BAO_predictions:
    z_BAO = np.loadtxt("f_Q_gravity_bao_data.txt", usecols=0, unpack=True)

    def comoving_distance_Hubble_distance_from_z(Omega_m, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0, z):
        """
        Calculate the comoving distance and Hubble distances in Mpc from redshift z using the Hubble distance.
        """
        h = H0 / 100.0
        comoving_distance_ = comoving_distance_vec(Omega_m, w_gamma/(h**2), w_nu/(h**2), w_nu_ur/(h**2), a_nr_sq, h*H_100, z)  # in Mpc
        Hubble_distance_ = c / (h*H_100*reduced_hubble_factor(z, Omega_m, w_gamma/(h**2), w_nu/(h**2), w_nu_ur/(h**2), a_nr_sq))  # DH in Mpc
        return comoving_distance_, Hubble_distance_  # Return in Mpc

    w_b_mean, Omega_m_mean, H0_mean = results_min_chi2.x

    z_general = np.arange(0.0, 5.001, 0.001)

    comoving_distance_general, Hubble_distance_general = comoving_distance_Hubble_distance_from_z(
        Omega_m_mean, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0_mean, z_general
        )

    comoving_distance_BAO, Hubble_distance_BAO = comoving_distance_Hubble_distance_from_z(
        Omega_m_mean, w_gamma, w_nu, w_nu_ur, a_nr_sq, H0_mean, z_BAO
        )
    
    w_bc_mean = Omega_m_mean * (H0_mean / 100.0)**2

    r_d = 147.05 * Mpc * (w_b_mean/0.02236)**-0.13 * (w_bc_mean/0.1432)**-0.23 * (N_eff/3.04)**-0.1 # Sound horizon at drag epoch in Mpc

    with open("output/fQ_gravity_LCDM_bao_predictions.txt", "w") as f:
        f.write("#BAO predictions for the LCDM model\n")
        f.write("# z, DM, DH, DV, DM/r_d, DH/r_d, DV/r_d\n")
        for z, DM, DH in zip(z_BAO, comoving_distance_BAO, Hubble_distance_BAO):
            DV = (DM**2 * DH * z)**(1/3)
            f.write(f"{z:.6f} {DM/Mpc:.6f} {DH/Mpc:.6f} {DV/Mpc:.6f} {DM/r_d:.6f} {DH/r_d:.6f} {DV/r_d:.6f}\n")

    with open("output/fQ_gravity_LCDM_general_predictions.txt", "w") as f:
        f.write("#General predictions for the LCDM model\n")
        f.write("# z, DM, DH, DV, DM/r_d, DH/r_d, DV/r_d\n")
        for z, DM, DH in zip(z_general, comoving_distance_general, Hubble_distance_general):
            DV = (DM**2 * DH * z)**(1/3)
            f.write(f"{z:.6f} {DM/Mpc:.6f} {DH/Mpc:.6f} {DV/Mpc:.6f} {DM/r_d:.6f} {DH/r_d:.6f} {DV/r_d:.6f}\n")

if compute_BAO_significance_data_space:
    print("Computing BAO significance from CMB+bbn+CC chains...")
    try:
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_CMB+bbn+CC_'ext'_z.npz"
            else:
                filename = f"dynesty_results_z0_CMB+bbn+CC_'ext'_{fixed_z0_value}.npz"
        else:
            filename = f"dynesty_results_z0_CMB+bbn+CC_'ext'.npz"
        data_cmb = np.load(filename)
        samples_cmb = data_cmb['samples']
        weights_cmb = data_cmb['weights']
    except FileNotFoundError:
        raise FileNotFoundError("CMB+bbn+CC Nested sampling results file not found. Please run nested sampling with CMB+bbn+CC data first.")
    
    BAO_predictions = np.zeros((samples_cmb.shape[0], Data_DESI.shape[0]))
    # Compute the BAO predictions for each sample in the CMB-only chains
    for i, theta in enumerate(samples_cmb):
        if fix_z0:
            z0 = fixed_z0_value
            w_b, omega_bc, H0 = theta
        else: 
            w_b, omega_bc, H0, z0 = theta
        T_cmb_mod = T_CMB_Firas * (1+z0)
        w_gamma, w_nu, w_nu_ur, a_nr_sq, rho_gamma, rho_nu, rho_nu_ur = functions_of_Tcmb(T_cmb_mod)
        h = H0 / 100.0
        w_bc = omega_bc * h**2
        w_b_tilde = w_b/(1+z0)**3
        w_bc_tilde = w_bc/(1+z0)**3
        d_drag = np.array([w_b_tilde, w_bc_tilde, z0]) - T_d_ref  # Use the parameters to compute d_drag
        T_drag = T_d_ref_CMB + T_d_gradient @ d_drag + 0.5 * d_drag @ T_d_hessian @ d_drag
        z_drag = T_drag / T_cmb_mod - 1
        a_drag = 1 / (1 + z_drag)

        rho_b = w_b * rho_crit_100
        rho_bc = w_bc * rho_crit_100

        R_b_gamma = rho_b / rho_gamma
        R_bc_gamma = rho_bc / rho_gamma
        R_0 = 3/4 * R_b_gamma  # Multiply by a for other epochs

        a_eq = (1 + R_nu_gamma) / R_bc_gamma
        z_eq = 1 / a_eq - 1
        # --- Ratios ---
        R_eq = R_0 * a_eq
        R_drag = R_0 * a_drag

        # --- Sound Horizon ---
        u = (np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_eq)) / (1 + np.sqrt(R_eq))
        r_d = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u)
        BAO_predictions[i] = model_predictions(z_DESI, omega_bc, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2, a_nr_sq, h*H_100, r_d, z0, True)
    
    # Compute the mean and covariance of the BAO predictions
    mean_BAO_predictions = np.average(BAO_predictions, axis=0, weights=weights_cmb)
    cov_BAO_predictions = np.cov(BAO_predictions.T, aweights=weights_cmb)
    correlation_matrix = np.zeros_like(cov_BAO_predictions)
    
    # Now compute the chi-squared difference between the DESI data and the CMB-only BAO predictions
    diff_BAO = Data_DESI - mean_BAO_predictions
    total_cov = Cov_DESI + cov_BAO_predictions
    inv_total_cov = np.linalg.inv(total_cov)

    chi2_BAO_significance = diff_BAO @ inv_total_cov @ diff_BAO
    print(f"Chi-squared difference for BAO significance from CMB-only chains: {chi2_BAO_significance:.4f}")
    #Compute the tension in units of sigma
    dof = len(Data_DESI) 
    print(f"Degrees of freedom: {dof}")
    from scipy.stats import chi2, norm
    p_value = chi2.sf(chi2_BAO_significance, dof)
    sigma_tension = norm.isf(p_value/2)
    print(f"BAO significance from CMB-only chains in observation space: {sigma_tension:.4f} sigma")
    print(f"P-value: {p_value:.4e}")

    # Diagnostic: Compare the "size" of the covariance matrices
    #trace_data = np.trace(Cov_DESI)
    #trace_model = np.trace(cov_BAO_predictions)

    #print(f"Trace of Data Covariance: {trace_data:.2e}")
    #print(f"Trace of Model Covariance: {trace_model:.2e}")
    #print(f"Ratio (Model/Data): {trace_model/trace_data:.2%}")

def sound_horizon_drag(w_b, w_bc, z0):
    T_cmb_mod = T_CMB_Firas * (1+z0)
    _, _, _, _, rho_gamma, _, _ = functions_of_Tcmb(T_cmb_mod)
    w_b_tilde = w_b/(1+z0)**3
    w_bc_tilde = w_bc/(1+z0)**3
    d_drag = np.array([w_b_tilde, w_bc_tilde, z0]) - T_d_ref  # Use the parameters to compute d_drag
    T_drag = T_d_ref_CMB + T_d_gradient @ d_drag + 0.5 * d_drag @ T_d_hessian @ d_drag
    z_drag = T_drag / T_cmb_mod - 1
    a_drag = 1 / (1 + z_drag)

    rho_b = w_b * rho_crit_100
    rho_bc = w_bc * rho_crit_100

    R_b_gamma = rho_b / rho_gamma
    R_bc_gamma = rho_bc / rho_gamma
    R_0 = 3/4 * R_b_gamma  # Multiply by a for other epochs

    a_eq = (1 + R_nu_gamma) / R_bc_gamma
    z_eq = 1 / a_eq - 1
    # --- Ratios ---
    R_eq = R_0 * a_eq
    R_drag = R_0 * a_drag

    # --- Sound Horizon ---
    u = (np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_eq)) / (1 + np.sqrt(R_eq))
    rd = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u)
    return rd

if compute_BAO_significance_parameter_space:
    print("Computing BAO significance from parameter space...")
    try:
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_CMB+bbn+CC_'ext'_z.npz"
            else:
                filename = f"dynesty_results_z0_CMB+bbn+CC_'ext'_{fixed_z0_value}.npz"
        else:
            filename = f"dynesty_results_z0_CMB+bbn+CC_'ext'.npz"
        data_cmb = np.load(filename)
        samples_cmb = data_cmb['samples']
        weights_cmb = data_cmb['weights']
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename2 = f"dynesty_results_LCDM_desi_only_'ext'_z.npz"
            else:
                filename2 = f"dynesty_results_z0_desi_only_'ext'_{fixed_z0_value}.npz"
        else:
            filename2 = f"dynesty_results_z0_desi_only_'ext'.npz"
        data_desi = np.load(filename2)
        samples_desi = data_desi['samples']
        weights_desi = data_desi['weights']
        
    except FileNotFoundError:
        raise FileNotFoundError("All-data Nested sampling results file not found. Please run nested sampling with all data first.")
    
    omega_m_values = samples_desi[:,0]
    H0_rd_values = samples_desi[:,1]
    omega_m_mean = np.average(omega_m_values, weights=weights_desi)
    H0_rd_mean = np.average(H0_rd_values, weights=weights_desi)
    covariance_omega_m_H0_rd_desi = np.cov(np.stack((omega_m_values, H0_rd_values)), aweights=weights_desi)
    #covariance_omega_m_H0_rd_desi = np.diag(np.diag(covariance_omega_m_H0_rd_desi))
    z0 = fixed_z0_value
    wb_values = samples_cmb[:,0]
    Omega_bc_values = samples_cmb[:,1]
    h_values = samples_cmb[:,2] / 100.0
    h_sq_values = h_values * h_values
    w_bc_values = Omega_bc_values * h_sq_values
    r_d = np.zeros(len(wb_values))
    for i in range(len(wb_values)):
        r_d[i] = sound_horizon_drag(wb_values[i], w_bc_values[i], z0)
    #print(r_d/Mpc)
    #sys.exit()

    H0_rd_cmb = 100.0*r_d*h_values/Mpc
    T_cmb_mod = T_CMB_Firas*(1+z0)
    _, w_nu, _, _, _, _, _ = functions_of_Tcmb(T_cmb_mod)
    omega_m_cmb = Omega_bc_values + w_nu/h_sq_values
    omega_m_cmb_mean = np.average(omega_m_cmb, weights=weights_cmb)
    #print(f"Omega_m CMB mean: {omega_m_cmb_mean:.5f}, DESI mean: {omega_m_mean:.5f}")
    H0_rd_cmb_mean = np.average(H0_rd_cmb, weights=weights_cmb)
    #print(f"H0*rd CMB mean: {H0_rd_cmb_mean:.3f}, DESI mean: {H0_rd_mean:.3f}")
    covariance_omega_m_H0_rd_cmb = np.cov(np.stack((omega_m_cmb, H0_rd_cmb)), aweights=weights_cmb)
    #covariance_omega_m_H0_rd_cmb = np.diag(np.diag(covariance_omega_m_H0_rd_cmb))

    mean_diff = np.array([omega_m_mean - omega_m_cmb_mean, H0_rd_mean - H0_rd_cmb_mean])
    total_covariance = covariance_omega_m_H0_rd_desi + covariance_omega_m_H0_rd_cmb
    mean_chi2_desi = mean_diff.T @ np.linalg.inv(total_covariance) @ mean_diff

    print(f"Mean Chi-squared for DESI from parameter space: {mean_chi2_desi:.4f}")
    from scipy.stats import chi2, norm
    dof = 2 
    p_value = chi2.sf(mean_chi2_desi, dof)
    sigma_tension = norm.isf(p_value/2)
    print(f"BAO significance from parameter space: {sigma_tension:.4f} sigma")
    print(f"P-value: {p_value:.4e}")

def weighted_percentile(data, weights, percentiles, axis=0):
    """
    Compute weighted percentiles along specified axis.
    
    Parameters:
    -----------
    data : array
        Data array
    weights : array
        Weights for each sample
    percentiles : array-like
        Percentiles to compute (0-100)
    axis : int
        Axis along which to compute percentiles
        
    Returns:
    --------
    Array of percentile values
    """
    if axis == 0:
        # For each column, compute weighted percentile
        result = np.zeros((len(percentiles), data.shape[1]))
        for i in range(data.shape[1]):
            sorted_indices = np.argsort(data[:, i])
            sorted_data = data[sorted_indices, i]
            sorted_weights = weights[sorted_indices]
            
            # Cumulative sum of weights
            cumsum = np.cumsum(sorted_weights)
            cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]
            
            # Interpolate to find percentile values
            for j, p in enumerate(percentiles):
                result[j, i] = np.interp(p / 100.0, cumsum, sorted_data)
        return result
    else:
        raise NotImplementedError("Only axis=0 is implemented")

if compute_cosmological_parameters:
    try:
        if fix_z0:
            if abs(fixed_z0_value) < 1e-8:
                filename = f"dynesty_results_LCDM_no_h0_'ext'_z.npz"
            else:
                filename = f"dynesty_results_z0_no_h0_'ext'_{fixed_z0_value}.npz"
        else:
            filename = f"dynesty_results_z0_no_h0_'ext'.npz"
        data = np.load(filename)
        samples = data['samples']
        weights = data['weights']
    except FileNotFoundError:
        raise FileNotFoundError("Nested sampling results file not found. Please run nested sampling first.")

    w_b_values = samples[:, 0]
    omega_m_values = samples[:, 1]
    H0_values = samples[:, 2]
    if not fix_z0:
        z0_values = samples[:, 3]
    else:
        z0_values = np.full_like(w_b_values, fixed_z0_value)

    # Compute weighted means
    w_b_mean = np.average(w_b_values, weights=weights)
    omega_m_mean = np.average(omega_m_values, weights=weights)
    H0_mean = np.average(H0_values, weights=weights)
    z0_mean = np.average(z0_values, weights=weights)

    # Compute weighted upper errors (15.87th and 84.13th percentiles)
    w_b_upper = weighted_percentile(w_b_values.reshape(-1, 1), weights, [84.13])[0, 0] - w_b_mean
    omega_m_upper = weighted_percentile(omega_m_values.reshape(-1, 1), weights, [84.13])[0, 0] - omega_m_mean
    H0_upper = weighted_percentile(H0_values.reshape(-1, 1), weights, [84.13])[0, 0] - H0_mean
    z0_upper = weighted_percentile(z0_values.reshape(-1, 1), weights, [84.13])[0, 0] - z0_mean

    # Compute weighted lower errors (15.87th and 84.13th percentiles)
    w_b_lower = w_b_mean - weighted_percentile(w_b_values.reshape(-1, 1), weights, [15.87])[0, 0]
    omega_m_lower = omega_m_mean - weighted_percentile(omega_m_values.reshape(-1, 1), weights, [15.87])[0, 0]
    H0_lower = H0_mean - weighted_percentile(H0_values.reshape(-1, 1), weights, [15.87])[0, 0]
    z0_lower = z0_mean - weighted_percentile(z0_values.reshape(-1, 1), weights, [15.87])[0, 0]

    # Print results
    print("Cosmological Parameters from Nested Sampling:")
    print(f"w_b: {w_b_mean:.7f} (+{w_b_upper:.7f}/-{w_b_lower:.7f})")
    print(f"omega_m: {omega_m_mean:.6f} (+{omega_m_upper:.6f}/-{omega_m_lower:.6f})")
    print(f"H0: {H0_mean:.5f} (+{H0_upper:.5f}/-{H0_lower:.5f})")
    if not fix_z0:
        print(f"z0: {z0_mean:.6f} (+{z0_upper:.6f}/-{z0_lower:.6f})")

# =============================================================================
# z0 REDSHIFT-BINNING TEST
# Fits z0 at 5 progressively lower redshift cutoffs to check for redshift
# dependence in the BAO offset.  A genuine gravitational/DE effect should
# be present at all redshifts; a low-z systematic would disappear at high-z.
#
# Usage: set compute_z0_redshift_binning = True at the top of this file.
# Does NOT require a pre-existing chain.  Uses scipy minimization.
# Note: always fits with z0 free regardless of the global fix_z0 flag.
# =============================================================================

def make_filtered_desi_data(z_min):
    """
    Return (z_arr, data_arr, C_inv_arr) keeping only DESI bins with z > z_min.
    The BGS bin at z=0.295 has 1 measurement (DV/rd); all others have 2 (DM/rd, DH/rd).
    """
    z_full = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
    data_sizes = [1, 2, 2, 2, 2, 2, 2]  # measurements per redshift bin

    # Select bins with z > z_min
    keep = z_full > z_min
    z_arr = z_full[keep]

    # Build index map into the full 13-element data/covariance vector
    idx_start = []
    pos = 0
    for sz in data_sizes:
        idx_start.append(pos)
        pos += sz

    selected_rows = []
    for i, flag in enumerate(keep):
        if flag:
            start = idx_start[i]
            for j in range(data_sizes[i]):
                selected_rows.append(start + j)

    data_arr = Data_DESI[selected_rows]
    Cov_sub = Cov_DESI[np.ix_(selected_rows, selected_rows)]
    C_inv_arr = np.linalg.inv(Cov_sub)
    return z_arr, data_arr, C_inv_arr


def minimize_chi2_with_data(z_arr, data_arr, C_inv_arr,
                             z0_fixed=None, use_cc=True, use_cmb=True, use_bbn=True):
    """
    Minimise chi-squared using a custom (filtered) DESI dataset.
    Always uses the full CMB / BBN / CC datasets.
    z0_fixed=None  → z0 is a free parameter (4-parameter fit)
    z0_fixed=float → z0 is held fixed at that value (3-parameter fit)
    Returns (params_array, chi2_value).
    """
    from scipy.optimize import minimize as sp_minimize

    def chi2_custom(theta):
        if z0_fixed is None:
            w_b, omega_m, H0, z0 = theta
        else:
            w_b, omega_m, H0 = theta
            z0 = z0_fixed

        h = H0 / 100.0
        w_bc = omega_m * h**2
        w_b_tilde = w_b / (1 + z0)**3
        w_bc_tilde = w_bc / (1 + z0)**3
        T_cmb_mod = T_CMB_Firas * (1 + z0)
        w_gamma_l, w_nu_l, w_nu_ur_l, a_nr_sq_l, rho_gamma_l, rho_nu_l, rho_nu_ur_l = functions_of_Tcmb(T_cmb_mod)

        chi2 = 0.0

        if use_cmb:
            chi2 += chi2_cmb(w_bc, w_b, H0, z0)

        if use_bbn:
            chi2 += ((w_b_tilde - w_b_bbn) / ew_b_bbn)**2

        # DESI (filtered dataset)
        d_drag_l = np.array([w_b_tilde, w_bc_tilde, z0]) - T_d_ref
        T_drag_l = T_d_ref_CMB + T_d_gradient @ d_drag_l + 0.5 * d_drag_l @ T_d_hessian @ d_drag_l
        z_drag_l = T_drag_l / T_cmb_mod - 1
        a_drag_l = 1 / (1 + z_drag_l)

        rho_b_l = w_b * rho_crit_100
        rho_bc_l = w_bc * rho_crit_100
        R_b_gamma_l = rho_b_l / rho_gamma_l
        R_bc_gamma_l = rho_bc_l / rho_gamma_l
        R_0_l = 3 / 4 * R_b_gamma_l
        a_eq_l = (1 + R_nu_gamma) / R_bc_gamma_l
        R_eq_l = R_0_l * a_eq_l
        R_drag_l = R_0_l * a_drag_l
        u_l = (np.sqrt(1 + R_drag_l) + np.sqrt(R_drag_l + R_eq_l)) / (1 + np.sqrt(R_eq_l))
        r_d_l = c / H_100 * 2 / 3 * np.sqrt(3 / w_bc * a_eq_l / R_eq_l) * np.log(u_l)

        preds = model_predictions(z_arr, omega_m,
                                  w_gamma_l / h**2, w_nu_l / h**2, w_nu_ur_l / h**2,
                                  a_nr_sq_l, h * H_100, r_d_l, z0,
                                  extended_z=(0.295 in z_arr))
        diff = preds - data_arr
        chi2 += diff @ C_inv_arr @ diff

        if use_cc:
            z_CC_cosmo = (z_CC - z0) / (1 + z0)
            H_z_pred = H0 * (1 + z0) * reduced_hubble_factor(
                z_CC_cosmo, omega_m,
                w_gamma_l / h**2, w_nu_l / h**2, w_nu_ur_l / h**2, a_nr_sq_l)
            chi2 += np.sum(((H_z_pred - H_z_CC) / sigma_H_z_CC)**2)

        return chi2 if np.isfinite(chi2) else 1e30

    if z0_fixed is None:
        x0 = [0.0224, 0.30, 67.5, 0.0]
        bounds = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0), (-0.1, 0.1)]
    else:
        x0 = [0.0224, 0.30, 67.5]
        bounds = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0)]

    res = sp_minimize(chi2_custom, x0, method='L-BFGS-B', bounds=bounds)
    return res.x, res.fun


if compute_z0_redshift_binning:
    print("=" * 70)
    print("z0 REDSHIFT-BINNING TEST")
    print("z0 fitted at progressively lower redshift cutoffs")
    print("=" * 70)

    z_min_cuts = [0.0, 0.295, 0.510, 0.706, 0.934]
    results_binning = []

    for z_cut in z_min_cuts:
        z_arr_f, data_arr_f, C_inv_f = make_filtered_desi_data(z_cut)
        n_bins = len(z_arr_f)
        params, chi2_val = minimize_chi2_with_data(
            z_arr_f, data_arr_f, C_inv_f,
            z0_fixed=None, use_cc=True, use_cmb=True, use_bbn=True)
        w_b_f, omega_m_f, H0_f, z0_f = params
        results_binning.append((z_cut, n_bins, z0_f, H0_f, chi2_val))
        print(f"  z_min={z_cut:.3f}  N_bins={n_bins:2d}  "
              f"z0={z0_f:+.5f}  H0={H0_f:.3f}  chi2={chi2_val:.3f}")

    print()
    print("Summary:")
    print(f"{'z_min':>8}  {'N_bins':>6}  {'z0_best':>10}  {'H0_best':>8}  {'chi2':>8}")
    for row in results_binning:
        print(f"{row[0]:8.3f}  {row[1]:6d}  {row[2]:+10.6f}  {row[3]:8.4f}  {row[4]:8.4f}")

    _arr = np.array(results_binning)
    np.savetxt("z0_redshift_binning.txt", _arr,
               header="z_min  N_bins  z0_best  H0_best  chi2",
               fmt="%.4f  %d  %+.6f  %.4f  %.4f")
    print(f"\n  Results saved to z0_redshift_binning.txt")
    print("=" * 70)


# =============================================================================
# z0–H0 POSTERIOR CORRELATION
# Loads an existing nested sampling chain and reports the Pearson correlation
# coefficient between the z0 and H0 posterior samples.
#
# Physical expectation:
#   Local void:       z0 and H0 ANTI-correlated (deeper void → larger z0 AND larger H0)
#   DE clustering:    z0 and H0 roughly uncorrelated
#   Bulk flow:        z0 and H0 positively correlated (both track expansion rate)
#
# Usage: set compute_z0_H0_correlation = True at the top of this file.
# Requires an existing free-z0 .npz chain: run nested sampling with fix_z0=False first.
# The chain used is determined by nested_sampling_data.
# =============================================================================

if compute_z0_H0_correlation:
    print("=" * 70)
    print("z0–H0 POSTERIOR CORRELATION")
    print("=" * 70)

    import os
    # For this analysis we need a free-z0 chain (fix_z0=False naming convention)
    _chain_file = f"dynesty_results_z0_{nested_sampling_data.replace('+', 'p')}_ext.npz"

    # Fallback: try common alternatives
    if not os.path.exists(_chain_file):
        for _candidate in [
            "dynesty_results_z0_no_h0_ext.npz",
            "dynesty_results_z0_ext.npz",
            "dynesty_results_z0_all_ext.npz",
        ]:
            if os.path.exists(_candidate):
                _chain_file = _candidate
                break

    try:
        _data = np.load(_chain_file)
        _samples = _data['samples']
        _weights = _data['weights']
        print(f"  Loaded chain: {_chain_file}  ({_samples.shape[0]} samples, "
              f"{_samples.shape[1]} params)")
    except FileNotFoundError:
        print(f"  Chain file not found: {_chain_file}")
        print(f"  Run nested sampling with fix_z0=False first.")
        _samples = None

    if _samples is not None and _samples.shape[1] >= 4:
        # Parameter order: [w_b, omega_m, H0, z0]
        _H0_samples = _samples[:, 2]
        _z0_samples = _samples[:, 3]

        _w = _weights / _weights.sum()
        _H0_mean = np.dot(_w, _H0_samples)
        _z0_mean = np.dot(_w, _z0_samples)
        _H0_std  = np.sqrt(np.dot(_w, (_H0_samples - _H0_mean)**2))
        _z0_std  = np.sqrt(np.dot(_w, (_z0_samples - _z0_mean)**2))
        _cov_H0_z0 = np.dot(_w, (_H0_samples - _H0_mean) * (_z0_samples - _z0_mean))
        _r = _cov_H0_z0 / (_H0_std * _z0_std)

        print(f"\n  Posterior means:")
        print(f"    H0  = {_H0_mean:.3f} ± {_H0_std:.3f} km/s/Mpc")
        print(f"    z0  = {_z0_mean:+.5f} ± {_z0_std:.5f}")
        print(f"\n  Pearson correlation coefficient r(z0, H0) = {_r:+.4f}")
        print(f"\n  Interpretation:")
        if _r < -0.3:
            print(f"    Anti-correlation (r = {_r:.3f}): consistent with LOCAL VOID scenario.")
            print(f"    Deeper void → larger z0 (more redshift) AND larger apparent H0.")
        elif abs(_r) < 0.3:
            print(f"    Weak/no correlation (r = {_r:.3f}): consistent with DARK ENERGY CLUSTERING")
            print(f"    or other mechanisms where z0 and H0 vary independently.")
        else:
            print(f"    Positive correlation (r = {_r:.3f}): consistent with systematic effects")
            print(f"    where both H0 and z0 track the same underlying parameter.")

        print(f"\n  Note: For the local void, the Hubble tension predicts H0_local > H0_CMB.")
        print(f"  If the void drives both effects, then larger z0 ↔ larger H0 → anti-correlation.")
        print(f"  (Anti-correlation here means z0 increases as H0 decreases, or vice versa.")
        print(f"   Sign depends on whether we define z0 from the BAO shift or from H0 excess.)")

        _z0_sigma = _z0_mean / _z0_std if _z0_std > 0 else 0.0
        print(f"\n  z0 significance: {_z0_mean:+.5f} / {_z0_std:.5f} = {_z0_sigma:+.2f}σ")
        if abs(_z0_sigma) < 1:
            print(f"  → z0 is consistent with zero at < 1σ.")
        elif _z0_mean < 0:
            print(f"  → z0 central value is NEGATIVE — opposite sign to local void prediction.")
            print(f"     HBK20 predicts z0 = +0.0084; data prefers z0 = {_z0_mean:+.5f}.")
            print(f"     This weakly disfavours the local void as the dominant mechanism.")

    print("=" * 70)
