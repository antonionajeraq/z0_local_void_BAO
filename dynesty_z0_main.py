import numpy as np
from scipy.linalg import cholesky, cho_factor, cho_solve
from scipy.linalg.blas import dtrmv, ddot
from scipy.integrate import quad
import sys
import os

# ── Control flags ──────────────────────────────────────────────────────────────
# Core run flags (env-var controllable for cluster runs)
Include_SNe             = os.environ.get("INCLUDE_SNE", "0") == "1"   # add DES-Dovekie SNe Ia
compute_minimum_chi2    = os.environ.get("COMPUTE_MIN_CHI2", "1") == "1"
perform_nested_sampling = os.environ.get("PERFORM_NS", "1") == "1"
nested_sampling_data    = os.environ.get("NESTED_DATA", "no_h0")
#   "all" | "CMB_only" | "no_desi" | "no_h0" | "CMB+bbn+CC" | "desi_only"
# Data at z ≤ z_min are excluded (no H0 anchor). 0.29 keeps the BGS BAO bin; 0.5 drops it.
z_min                   = float(os.environ.get("Z_MIN", 0.29))
fix_z0                  = os.environ.get("FIX_Z0", "0") == "1"   # LCDM comparison: pin z0=0
fixed_z0_value          = float(os.environ.get("FIXED_Z0_VALUE", 0.0))  # value of z0 when fix_z0

# Post-processing analysis blocks (default OFF; run locally on existing chains)
compute_z0_redshift_binning              = os.environ.get("COMPUTE_Z0_BINNING",     "0") == "1"
compute_z0_H0_correlation                = os.environ.get("COMPUTE_Z0_H0_CORR",     "0") == "1"
compute_BAO_significance_data_space      = os.environ.get("COMPUTE_BAO_SIG_DATA",   "0") == "1"
compute_BAO_significance_parameter_space = os.environ.get("COMPUTE_BAO_SIG_PARAM",  "0") == "1"
compute_Universe_age                     = os.environ.get("COMPUTE_AGE",            "0") == "1"
compute_cosmological_parameters          = os.environ.get("COMPUTE_COSMO_PARAMS",   "0") == "1"

nprocs = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

# ── Physical constants ─────────────────────────────────────────────────────────
Sixth = 1 / 6
G = 6.6743e-11
c = 299792458
h_Planck = 6.62607e-34
hbar = h_Planck / (2 * np.pi)
k_Boltzmann = 1.380649e-23
eV = 1.602176634e-19
AU = 149597870700
Radian = 180 / np.pi
Degree = np.pi / 180
Arcsec = Degree / 3600
pc = AU / Arcsec
Mpc = pc * 1e6
H_100 = 100e3 / Mpc
H0_to_SI = 1e3 / Mpc

GM_Sun = 1.32712440041279419e20
Year = 2 * np.pi * np.sqrt(AU**3 / GM_Sun)
Gyr = Year * 1e9

# ── Cosmological parameters ────────────────────────────────────────────────────
# Neutrino sector (borrowed from Dovekie_pCPL/dynesty_sn_4nodes.py):
# three species with masses 50/10/0 meV (descending; Σm_ν = 0.06 eV). Species C is
# massless (N_nu_massless = 1). Each massive species transitions relativistic→matter
# with its OWN a_nr (knu = 2 ⇒ "_sq" form √(1+(a/a_nr)²)); the massless species stays
# radiation (→ +1 in the neutrino sum) and its present-day relativistic density is
# removed from Ω_Λ so that E²(z=0) = 1.
N_eff = 3.044  # effective neutrino number
Third = 1.0 / 3.0
mnu_A = 0.05 * eV / c**2   # neutrino rest masses (descending), scheme 50/10/0 meV
mnu_B = 0.01 * eV / c**2
mnu_C = 0.0  * eV / c**2   # massless
Sum_mnu = mnu_A + mnu_B + mnu_C
N_nu_massless = 1          # C massless: its relativistic density today reduces Ω_Λ
R_nu_gamma = N_eff * 7/8 * (4/11)**(4/3)

T_CMB_Firas = 2.72548
rho_crit_100 = 3 * H_100**2 / (8 * np.pi * G)

def functions_of_Tcmb(T_CMB):
    """Photon / neutrino densities at CMB temperature T_CMB.
    Returns (w_gamma, w_nu, w_nu_ur, a_nr_pair, rho_gamma, rho_nu, rho_nu_ur) where
    a_nr_pair = (a_nr_A_sqinv, a_nr_B_sqinv) are the per-species relativistic→matter
    transition descriptors for the two MASSIVE species (C is massless). rho_nu is the
    total massive density (A+B); w_nu_ur is the total ur-equivalent for 3 species."""
    rho_gamma = np.pi**2 / 15 * (k_Boltzmann**4) / (hbar**3 * c**5) * T_CMB**4
    zeta_3 = 1.202056903159594
    n_gamma = 16 * np.pi * zeta_3 * (k_Boltzmann * T_CMB / (h_Planck * c))**3
    n_nu_i = 3/11 * n_gamma * (Third * N_eff)**0.68   # per-species number density
    rho_nu_A = n_nu_i * mnu_A
    rho_nu_B = n_nu_i * mnu_B
    rho_nu = rho_nu_A + rho_nu_B                       # total massive (C massless → 0)
    rho_nu_ur = rho_gamma * R_nu_gamma                 # total ur-equivalent (3 species)
    rho_nu_ur_i = Third * rho_nu_ur                    # per-species ur-equivalent
    a_nr_A_sqinv = (rho_nu_A / rho_nu_ur_i)**2 - 1
    a_nr_B_sqinv = (rho_nu_B / rho_nu_ur_i)**2 - 1
    a_nr_pair = (a_nr_A_sqinv, a_nr_B_sqinv)
    w_gamma = rho_gamma / rho_crit_100
    w_nu    = rho_nu    / rho_crit_100
    w_nu_ur = rho_nu_ur / rho_crit_100
    return w_gamma, w_nu, w_nu_ur, a_nr_pair, rho_gamma, rho_nu, rho_nu_ur

# ── Taylor-expansion coefficients for CMB theta_star / r_drag ─────────────────
T_d_ref_CMB  = 2892.7189265390
T_d_ref      = np.array([0.02250, 0.1450, 0.01])
T_d_gradient = np.array([5926.7982685835, 203.2921847528, -2.3736921657])
T_d_hessian  = np.array([[-189596.6848072170, -4765.7156601812,  108.8993400754],
                          [  -4765.7156601812,  -624.6877708911,   -5.3358430973],
                          [   108.8993400754,     -5.3358430973,   17.1094164296]])

T_star_ref_CMB  = 2970.2490798866
T_star_ref      = np.array([0.02250, 0.1450, 0.01])
T_star_gradient = np.array([-3187.3522386402, 194.0022355988, -3.8997457516])
T_star_hessian  = np.array([[265184.9625445241, -7854.3444118611, 144.6125754140],
                             [ -7854.3444118611,  -491.8071934238,  -9.3247907229],
                             [  144.6125754140,     -9.3247907229,  23.4632437267]])

# ── DESI DR2 BAO data (full set, filtered at load time by z_min) ───────────────
_z_DESI_full = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
_data_DESI_full = np.array([7.942,
                             13.588, 21.863,
                             17.351, 19.455,
                             21.576, 17.641,
                             27.601, 14.176,
                             30.512, 12.817,
                             38.988,  8.632])
_C_blocks_full = [
    0.075**2,
    np.array([[0.167**2, -0.459*0.167*0.425], [-0.459*0.167*0.425, 0.425**2]]),
    np.array([[0.177**2, -0.404*0.177*0.330], [-0.404*0.177*0.330, 0.330**2]]),
    np.array([[0.152**2, -0.416*0.152*0.193], [-0.416*0.152*0.193, 0.193**2]]),
    np.array([[0.318**2, -0.434*0.318*0.221], [-0.434*0.318*0.221, 0.221**2]]),
    np.array([[0.760**2, -0.500*0.760*0.516], [-0.500*0.760*0.516, 0.516**2]]),
    np.array([[0.531**2, -0.431*0.531*0.101], [-0.431*0.531*0.101, 0.101**2]])]
_data_sizes_full = [1, 2, 2, 2, 2, 2, 2]  # 1 = DV/rd (BGS); 2 = DM/rd, DH/rd

def _filter_desi(z_min_cut):
    mask = _z_DESI_full > z_min_cut
    z_sel    = _z_DESI_full[mask]
    blocks   = [b for b, m in zip(_C_blocks_full,   mask) if m]
    sizes    = [s for s, m in zip(_data_sizes_full, mask) if m]
    ptr, parts = 0, []
    for size, inc in zip(_data_sizes_full, mask):
        if inc:
            parts.append(_data_DESI_full[ptr:ptr+size])
        ptr += size
    data_sel = np.concatenate(parts)
    n = sum(sizes)
    Cov = np.zeros((n, n))
    row = 0
    for blk in blocks:
        if np.isscalar(blk):
            Cov[row, row] = blk; row += 1
        else:
            s = blk.shape[0]; Cov[row:row+s, row:row+s] = blk; row += s
    C_inv = np.linalg.inv(Cov)
    return z_sel, data_sel, C_inv, sizes

z_DESI, Data_DESI, C_inv_total, _desi_sizes = _filter_desi(z_min)
C_inv_total_LC = np.asfortranarray(cholesky(C_inv_total, lower=True, check_finite=False))
logdet_DESI = len(Data_DESI) * np.log(2 * np.pi) - 2 * np.sum(np.log(np.diag(C_inv_total_LC)))
print(f"DESI bins used (z > {z_min}): {len(z_DESI)}")

# ── Cosmic Chronometer data ────────────────────────────────────────────────────
_z_CC_all, _H_CC_all, _sH_CC_all = np.loadtxt('CC_data.txt', unpack=True)
_cc_mask     = _z_CC_all > z_min
z_CC         = _z_CC_all[_cc_mask]
H_z_CC       = _H_CC_all[_cc_mask]
sigma_H_z_CC = _sH_CC_all[_cc_mask]
logdet_CC    = np.sum(np.log(2 * np.pi * sigma_H_z_CC**2))
print(f"CC data points used (z > {z_min}): {len(z_CC)}")

# ── SN data (loaded only when Include_SNe): quality cuts + z_min cut via Schur ──
def _Mask_Schur_complement(M, flag):
    flag_excl = ~flag
    B12 = M[np.ix_(flag, flag_excl)]
    B22 = M[np.ix_(flag_excl, flag_excl)]
    u, lower = cho_factor(B22)
    X = cho_solve((u, lower), B12.T)
    R = M[np.ix_(flag, flag)] - B12 @ X
    return 0.5 * (R + R.T)

if Include_SNe:
    import pandas as pd
    Min_ProbIA = 0.9
    if not os.path.exists("DES-Dovekie_HD.csv"):
        print("File not found: DES-Dovekie_HD.csv"); sys.exit(0)
    _df = pd.read_csv("DES-Dovekie_HD.csv", skiprows=8, sep=r'\s+')
    _IDSURVEY = _df['IDSURVEY'].astype(int).values
    _zHD      = _df['zHD'].values
    _zHEL     = _df['zHEL'].values
    _MU       = _df['MU'].values
    _PROBIA   = _df['PROBIA_BEAMS'].values

    if not os.path.exists('STAT+SYS.npz'):
        print("File not found: STAT+SYS.npz"); sys.exit(0)
    _sn_raw = np.load('STAT+SYS.npz')
    _nsn = _sn_raw['nsn'][0]
    _C_full = np.zeros((_nsn, _nsn))
    _C_full[np.triu_indices(_nsn)] = _sn_raw['cov']
    _C_full = _C_full + _C_full.T - np.diag(np.diag(_C_full))

    Flag_SN = (_IDSURVEY == 10) & (_PROBIA > Min_ProbIA) & (_zHD > z_min)
    DES_cost_matrix = _Mask_Schur_complement(_C_full, Flag_SN)
    _sort = np.argsort(_zHD[Flag_SN])
    DES_cost_matrix = DES_cost_matrix[_sort, :][:, _sort]
    zHD  = _zHD[Flag_SN][_sort]
    zHEL = _zHEL[Flag_SN][_sort]
    MU   = _MU[Flag_SN][_sort]
    N_SNe = len(zHD)
    print(f"SNe used (IDSURVEY=10, P_Ia>{Min_ProbIA}, z>{z_min}): {N_SNe}")
    DES_cost_matrix_LC = np.asfortranarray(cholesky(DES_cost_matrix, lower=True, check_finite=False))
    logdet_SN = N_SNe * np.log(2 * np.pi) - 2 * np.sum(np.log(np.diag(DES_cost_matrix_LC)))
else:
    N_SNe = 0

# ── Pre-built Simpson grids in z_cosmo space ───────────────────────────────────
# z_cosmo = (z_obs - z0)/(1+z0). For z0 ∈ [−0.1, 0.1], max z_cosmo occurs at z0 = −0.1.
_z0_buf = 0.12
_z_bao_grid_max = (_z_DESI_full[-1] + _z0_buf) / (1 - _z0_buf)

N_grid_bao = 2000
z_grid_bao = np.linspace(0.0, _z_bao_grid_max, N_grid_bao)
dz_bao     = np.diff(z_grid_bao)
z_mid_bao  = 0.5 * (z_grid_bao[:-1] + z_grid_bao[1:])
Sixth_dz_bao = dz_bao / 6.0
z_interleaved_bao = np.empty(2 * N_grid_bao - 1)
z_interleaved_bao[::2]  = z_grid_bao
z_interleaved_bao[1::2] = z_mid_bao

if Include_SNe and N_SNe > 0:
    _z_sn_grid_max = (np.max(zHD) + _z0_buf) / (1 - _z0_buf)
    N_grid_sn = 2000
    z_grid_sn = np.linspace(0.0, _z_sn_grid_max, N_grid_sn)
    dz_sn     = np.diff(z_grid_sn)
    z_mid_sn  = 0.5 * (z_grid_sn[:-1] + z_grid_sn[1:])
    Sixth_dz_sn = dz_sn / 6.0
    z_interleaved_sn = np.empty(2 * N_grid_sn - 1)
    z_interleaved_sn[::2]  = z_grid_sn
    z_interleaved_sn[1::2] = z_mid_sn

# ── Hubble factor (2 massive + 1 massless neutrino) ────────────────────────────
def reduced_hubble_factor(z, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair,
                          Massive_Neutrinos=False):
    """ΛCDM reduced Hubble factor E(z) with the 50/10/0 meV neutrino scheme.
    Massive_Neutrinos=True : the two massive species count as matter (Ω_nu·a⁻³),
                             valid for z ≪ z_nr ~ 100 (BAO/SN/CC).
    Massive_Neutrinos=False: full per-species relativistic→matter transition
                             (√(1+(a/a_nr)²) per massive species, +1 for the massless
                             one); use only for CMB quantities (z ~ 1100).
    Ω_Λ subtracts the present-day massless-neutrino relativistic density so E²(0)=1
    (the Massive_Neutrinos=True low-z branch is left ~1e-5 off by design)."""
    a_nr_A_sqinv, a_nr_B_sqinv = a_nr_pair
    Omega_nu_ur_i = Omega_nu_ur * Third
    Omega_Lambda = 1.0 - Omega_bc - Omega_gamma - Omega_nu - N_nu_massless * Omega_nu_ur_i
    a_inv = 1.0 + np.asarray(z)
    if Massive_Neutrinos:
        E2 = (Omega_bc + Omega_nu) * a_inv**3 + Omega_gamma * a_inv**4 + Omega_Lambda
    else:
        a_sq = 1.0 / a_inv**2
        nu_sum = (np.sqrt(1.0 + a_sq * a_nr_A_sqinv)
                  + np.sqrt(1.0 + a_sq * a_nr_B_sqinv) + 1.0)
        E2 = (Omega_bc * a_inv**3 + Omega_gamma * a_inv**4 + Omega_Lambda
              + Omega_nu_ur_i * nu_sum * a_inv**4)
    return np.sqrt(E2)

def _dc_simpson(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair, H0_SI,
                z_values, z_grid, z_interleaved, Sixth_dz):
    """Vectorized comoving distance (metres) at z_values via pre-built Simpson grid."""
    E_all  = reduced_hubble_factor(z_interleaved, Omega_bc, Omega_gamma, Omega_nu,
                                   Omega_nu_ur, a_nr_pair, Massive_Neutrinos=True)
    E_grid = E_all[::2]
    E_mid  = E_all[1::2]
    inv_H  = c / H0_SI
    dc_steps = inv_H * Sixth_dz * (1.0/E_grid[:-1] + 4.0/E_mid + 1.0/E_grid[1:])
    dc_cumul = np.concatenate([[0.0], np.cumsum(dc_steps)])
    return np.interp(z_values, z_grid, dc_cumul)

def _dc_arcsinh(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair, H0_SI,
                z_star, N_grid=4000):
    """Comoving distance (metres) to z_star via arcsinh-transformed quadrature.
    Full neutrino treatment — only for CMB theta_star (z ~ 1000)."""
    u_grid = np.linspace(0.0, np.arcsinh(z_star), N_grid)
    z_g    = np.sinh(u_grid)
    E_g    = reduced_hubble_factor(z_g, Omega_bc, Omega_gamma, Omega_nu,
                                   Omega_nu_ur, a_nr_pair, Massive_Neutrinos=False)
    return c / H0_SI * np.trapezoid(np.cosh(u_grid) / E_g, u_grid)

# ── CMB shift parameters (Camphuis et al. 2026) ───────────────────────────────
mu_cmb     = np.array([1.04161119, 0.02239789, 0.14265263])  # first chain not converged as 41 autocorr times < 50
cov_cmb    = 1.0e-9 * np.array([[ 53.84815397,   1.1072339,  -28.70053592],
                                  [  1.1072339,    9.20446757, -17.60162507],
                                  [-28.70053592, -17.60162507, 843.13219472]])  # other 7 chains, 3 tau_f removed each (~0.1 of chain)
inv_cov_cmb = np.linalg.inv(cov_cmb)
inv_cov_cmb_LC = np.asfortranarray(cholesky(inv_cov_cmb, lower=True, check_finite=False))
logdet_cmb = 3 * np.log(2 * np.pi) - 2 * np.sum(np.log(np.diag(inv_cov_cmb_LC)))

def _cmb_quantities(w_b, w_bc, z0):
    """Return (r_rec, r_d, z_rec, w_gamma, w_nu, w_nu_ur, a_nr_pair) for the z0 shift.
    Sound horizons are analytic; DM_rec (for theta*) is computed separately.
    Densities returned here are evaluated at the shifted CMB temperature T_CMB·(1+z0)
    and are reused for the low-z (BAO/CC/SN) terms for full internal consistency."""
    w_b_tilde  = w_b  / (1 + z0)**3
    w_bc_tilde = w_bc / (1 + z0)**3
    T_cmb_mod  = T_CMB_Firas * (1 + z0)
    w_gamma, w_nu, w_nu_ur, a_nr_pair, rho_gamma, _, _ = functions_of_Tcmb(T_cmb_mod)
    rho_b  = w_b  * rho_crit_100
    rho_bc = w_bc * rho_crit_100

    d_drag = np.array([w_b_tilde, w_bc_tilde, z0]) - T_d_ref
    T_drag = T_d_ref_CMB + T_d_gradient @ d_drag + 0.5 * d_drag @ T_d_hessian @ d_drag
    z_drag = T_drag / T_cmb_mod - 1
    a_drag = 1 / (1 + z_drag)

    d_rec = np.array([w_b_tilde, w_bc_tilde, z0]) - T_star_ref
    T_rec = T_star_ref_CMB + T_star_gradient @ d_rec + 0.5 * d_rec @ T_star_hessian @ d_rec
    z_rec = T_rec / T_cmb_mod - 1
    a_rec = 1 / (1 + z_rec)

    R_b_gamma  = rho_b  / rho_gamma
    R_bc_gamma = rho_bc / rho_gamma
    R_0   = 3/4 * R_b_gamma
    a_eq  = (1 + R_nu_gamma) / R_bc_gamma
    R_eq  = R_0 * a_eq

    R_rec = R_0 * a_rec
    u_rec = (np.sqrt(1 + R_rec) + np.sqrt(R_rec + R_eq)) / (1 + np.sqrt(R_eq))
    r_rec = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u_rec)

    R_drag = R_0 * a_drag
    u_drag = (np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_eq)) / (1 + np.sqrt(R_eq))
    r_d    = c / H_100 * 2/3 * np.sqrt(3 / w_bc * a_eq / R_eq) * np.log(u_drag)

    return r_rec, r_d, z_rec, w_gamma, w_nu, w_nu_ur, a_nr_pair

def theta_star_and_rd(w_b, w_bc, h, z0):
    r_rec, r_d, z_rec, w_gamma, w_nu, w_nu_ur, a_nr_pair = _cmb_quantities(w_b, w_bc, z0)
    DM_rec = _dc_arcsinh(w_bc/h**2, w_gamma/h**2, w_nu/h**2, w_nu_ur/h**2,
                         a_nr_pair, h * H_100, z_rec)
    theta_s = r_rec / DM_rec
    return theta_s, r_d

def chi2_cmb(w_bc, w_b, H0, z0):
    h = H0 / 100.0
    theta_s, _ = theta_star_and_rd(w_b, w_bc, h, z0)
    r = np.array([100.0 * theta_s, w_b/(1+z0)**3, w_bc/(1+z0)**3]) - mu_cmb
    dtrmv(inv_cov_cmb_LC, r, lower=1, trans=1, overwrite_x=1)
    return ddot(r, r)

# ── BAO model predictions with z0 mapping ─────────────────────────────────────
def model_predictions(z_obs, Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur,
                      a_nr_pair, H0_SI, r_d, z0, sizes=None):
    """BAO observables at observed redshifts z_obs, accounting for the z0 shift.
    Returns the data vector matching the `sizes` layout (DV/rd or DM/rd,DH/rd per bin).
    sizes defaults to the global filtered DESI layout."""
    if sizes is None:
        sizes = _desi_sizes
    z_cosmo = (z_obs - z0) / (1 + z0)
    DM = _dc_simpson(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair, H0_SI,
                     z_cosmo, z_grid_bao, z_interleaved_bao, Sixth_dz_bao)
    E_cosmo = reduced_hubble_factor(z_cosmo, Omega_bc, Omega_gamma, Omega_nu,
                                    Omega_nu_ur, a_nr_pair, Massive_Neutrinos=True)
    DH = c / (H0_SI * (1 + z0) * E_cosmo)
    out = []
    for i, size in enumerate(sizes):
        if size == 1:
            DV = (DM[i]**2 * DH[i] * z_obs[i])**(1.0/3.0)
            out.append(DV / r_d)
        else:
            out.extend([DM[i] / r_d, DH[i] / r_d])
    return np.array(out)

c_km_s = c / 1000.0
def model_predictions_desi_only(omega_m, H0rd, z0, z_obs=None, sizes=None):
    """Calibration-free DESI predictions parametrised by H0rd ≡ H0[km/s/Mpc]·r_d[Mpc].
    Pure matter+Λ (neutrinos negligible), dimensionless output."""
    if z_obs is None:
        z_obs = z_DESI
    if sizes is None:
        sizes = _desi_sizes
    z_cosmo = (z_obs - z0) / (1 + z0)
    zero_pair = (0.0, 0.0)
    E_all = reduced_hubble_factor(z_interleaved_bao, omega_m, 0.0, 0.0, 0.0, zero_pair,
                                  Massive_Neutrinos=True)
    E_grid = E_all[::2]; E_mid = E_all[1::2]
    steps = Sixth_dz_bao * (1.0/E_grid[:-1] + 4.0/E_mid + 1.0/E_grid[1:])
    cumul = np.concatenate([[0.0], np.cumsum(steps)])
    integral = np.interp(z_cosmo, z_grid_bao, cumul)
    pref = c_km_s / H0rd
    DM = pref * integral                                  # = D_M / r_d
    Ez = np.interp(z_cosmo, z_grid_bao, E_grid)
    DH = pref / Ez / (1 + z0)                             # = D_H / r_d
    out = []
    for i, size in enumerate(sizes):
        if size == 1:
            DV = (DM[i]**2 * DH[i] * z_obs[i])**(1.0/3.0)
            out.append(DV)
        else:
            out.extend([DM[i], DH[i]])
    return np.array(out)

# ── SN chi-squared with z0 mapping ────────────────────────────────────────────
def chisq_SN(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair, H0_SI, z0, M_B):
    """SN Ia chi-squared. z0 maps to the cosmological redshift for the comoving
    distance; the heliocentric redshift (in the (1+z) luminosity-distance factor) is a
    direct measurement and is NOT shifted."""
    z_cosmo_SN = (zHD - z0) / (1 + z0)
    Dc = _dc_simpson(Omega_bc, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair, H0_SI,
                     z_cosmo_SN, z_grid_sn, z_interleaved_sn, Sixth_dz_sn)
    DL = Dc / Mpc * (1.0 + zHEL)
    mu_model = 5.0 * np.log10(DL) + 25.0
    residuals = MU - (M_B + mu_model)
    dtrmv(DES_cost_matrix_LC, residuals, lower=1, trans=1, overwrite_x=1)
    return ddot(residuals, residuals)

# ── Priors and external datasets ───────────────────────────────────────────────
w_b_bbn  = 0.021915   # BBN baryon density: 100 w_b = 2.1915 ± 0.0215, ApSS (2026) 371:20
ew_b_bbn = 0.000215   # https://doi.org/10.1103/wspy-s948 table 1 (BBN only, PRIMAT fixed N_eff)
logdet_BBN = np.log(2 * np.pi * ew_b_bbn**2)

H0_SHOES = 73.17;  eH0_SHOES = 0.86
H0_SNII  = 74.9;   eH0_SNII  = 2.7
H0_SBF   = 73.8;   eH0_SBF   = 2.4
H0_maser = 73.9;   eH0_maser = 3.0
logdet_H0 = (np.log(2*np.pi*eH0_SHOES**2) + np.log(2*np.pi*eH0_SNII**2)
           + np.log(2*np.pi*eH0_SBF**2)   + np.log(2*np.pi*eH0_maser**2))

# ── Parameter layout helpers ───────────────────────────────────────────────────
# Standard layout:  [w_b, omega_m, H0, (z0), (M_B)]   (z0 dropped if fix_z0; M_B if not Include_SNe)
# desi_only layout:  [omega_m, H0rd, (z0)]
def _unpack(theta):
    w_b = theta[0]; omega_m = theta[1]; H0 = theta[2]
    idx = 3
    if fix_z0:
        z0 = fixed_z0_value
    else:
        z0 = theta[idx]; idx += 1
    M_B = theta[idx] if Include_SNe else None
    return w_b, omega_m, H0, z0, M_B

def log_prior(theta):
    if nested_sampling_data == "desi_only":
        if fix_z0:
            omega_m, H0rd = theta; z0 = fixed_z0_value
        else:
            omega_m, H0rd, z0 = theta
            if not (-0.10 < z0 < 0.10): return -np.inf
        if not (0.00 < omega_m < 0.50): return -np.inf
        if not (5000.0 < H0rd < 13000.0): return -np.inf
        return 0.0
    w_b, omega_m, H0, z0, M_B = _unpack(theta)
    if not (0.020 < w_b     < 0.025): return -np.inf
    if not (0.00  < omega_m < 0.50):  return -np.inf
    if not (50.0  < H0      < 100.0): return -np.inf
    if not fix_z0 and not (-0.10 < z0 < 0.10): return -np.inf
    if Include_SNe and not (-50.0 < M_B < 50.0): return -np.inf
    return 0.0

# ── Core chi-squared assembly ─────────────────────────────────────────────────
def _chi2_core(w_b, omega_m, H0, z0, M_B,
               use_desi, use_h0, use_cc, use_cmb, use_bbn, use_sn,
               want_parts=False):
    h     = H0 / 100.0
    w_bc  = omega_m * h**2
    H0_SI = H0 * H0_to_SI

    # CMB quantities + (z0-shifted) densities, reused for all low-z terms.
    r_rec, r_d, z_rec, w_gamma, w_nu, w_nu_ur, a_nr_pair = _cmb_quantities(w_b, w_bc, z0)
    Omega_gamma = w_gamma / h**2
    Omega_nu    = w_nu    / h**2
    Omega_nu_ur = w_nu_ur / h**2

    chi2 = 0.0; logd = 0.0
    parts = {}

    if use_cmb:
        DM_rec  = _dc_arcsinh(omega_m, Omega_gamma, Omega_nu, Omega_nu_ur,
                              a_nr_pair, h * H_100, z_rec)
        theta_s = r_rec / DM_rec
        r = np.array([100.0 * theta_s, w_b/(1+z0)**3, w_bc/(1+z0)**3]) - mu_cmb
        dtrmv(inv_cov_cmb_LC, r, lower=1, trans=1, overwrite_x=1)
        c_cmb = ddot(r, r); chi2 += c_cmb; logd += logdet_cmb; parts['CMB'] = c_cmb

    if use_bbn:
        c_bbn = ((w_b/(1+z0)**3 - w_b_bbn) / ew_b_bbn)**2
        chi2 += c_bbn; logd += logdet_BBN; parts['BBN'] = c_bbn

    if use_desi:
        preds = model_predictions(z_DESI, omega_m, Omega_gamma, Omega_nu, Omega_nu_ur,
                                  a_nr_pair, H0_SI, r_d, z0)
        diff = preds - Data_DESI
        dtrmv(C_inv_total_LC, diff, lower=1, trans=1, overwrite_x=1)
        c_desi = ddot(diff, diff); chi2 += c_desi; logd += logdet_DESI; parts['DESI'] = c_desi

    if use_h0:
        c_h0 = (((H0 - H0_SHOES) / eH0_SHOES)**2 + ((H0 - H0_SNII) / eH0_SNII)**2
              + ((H0 - H0_SBF)   / eH0_SBF  )**2 + ((H0 - H0_maser) / eH0_maser)**2)
        chi2 += c_h0; logd += logdet_H0; parts['H0'] = c_h0

    if use_cc:
        z_CC_cosmo = (z_CC - z0) / (1 + z0)
        # Correct z0 mapping (referee fix): H_CC^obs(z_obs) = H(z_c), z_c=(z_obs-z0)/(1+z0).
        # No extra (1+z0) prefactor — the (1+z0) Jacobian in dz cancels the (1+z)
        # denominator of the CC observable, leaving H_pred = H0·E(z_c).
        H_pred = H0 * reduced_hubble_factor(z_CC_cosmo, omega_m, Omega_gamma, Omega_nu,
                                            Omega_nu_ur, a_nr_pair, Massive_Neutrinos=True)
        c_cc = np.sum(((H_pred - H_z_CC) / sigma_H_z_CC)**2)
        chi2 += c_cc; logd += logdet_CC; parts['CC'] = c_cc

    if use_sn:
        c_sn = chisq_SN(omega_m, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair,
                        H0_SI, z0, M_B)
        chi2 += c_sn; logd += logdet_SN; parts['SN'] = c_sn

    if want_parts:
        return chi2, logd, parts
    return chi2, logd

# ── Likelihood ─────────────────────────────────────────────────────────────────
def log_likelihood(theta, use_desi=True, use_h0=False, use_cc=True,
                   use_cmb=True, use_bbn=True, use_sn=False):
    if nested_sampling_data == "desi_only":
        if fix_z0:
            omega_m, H0rd = theta; z0 = fixed_z0_value
        else:
            omega_m, H0rd, z0 = theta
        preds = model_predictions_desi_only(omega_m, H0rd, z0)
        diff = preds - Data_DESI
        _d = diff.copy()
        dtrmv(C_inv_total_LC, _d, lower=1, trans=1, overwrite_x=1)
        chi2 = ddot(_d, _d)
        if not np.isfinite(chi2): return -np.inf
        return -0.5 * (chi2 + logdet_DESI)

    w_b, omega_m, H0, z0, M_B = _unpack(theta)
    chi2, logd = _chi2_core(w_b, omega_m, H0, z0, M_B,
                            use_desi, use_h0, use_cc, use_cmb, use_bbn, use_sn)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * (chi2 + logd)

def chi_squared(theta, use_desi=True, use_h0=False, use_cc=True, use_cmb=True,
                use_bbn=True, use_sn=None, print_output=False):
    """Full-theta chi-squared (z0 always explicit). Layout:
    Include_SNe → [w_b, omega_m, H0, z0, M_B]; else [w_b, omega_m, H0, z0]."""
    if use_sn is None:
        use_sn = Include_SNe
    if Include_SNe:
        w_b, omega_m, H0, z0, M_B = theta
    else:
        w_b, omega_m, H0, z0 = theta; M_B = None
    chi2, logd, parts = _chi2_core(w_b, omega_m, H0, z0, M_B,
                                   use_desi, use_h0, use_cc, use_cmb, use_bbn, use_sn,
                                   want_parts=True)
    if print_output:
        print("Chi2 contributions:")
        for key in ("CMB", "BBN", "DESI", "H0", "CC", "SN"):
            if key in parts:
                print(f"  {key}: {parts[key]:.3f}")
        print(f"Total Chi2: {chi2:.3f}")
    if not np.isfinite(chi2):
        return 1e10
    return chi2

# ── Best-fit minimization (sequential: z0 fixed, then free) ────────────────────
def _theta_full(reduced, z0_val):
    """Insert z0 into a (z0-free) reduced parameter list at the right slot."""
    if Include_SNe:
        w_b, omega_m, H0, M_B = reduced
        return [w_b, omega_m, H0, z0_val, M_B]
    w_b, omega_m, H0 = reduced
    return [w_b, omega_m, H0, z0_val]

def minimize_chi2_sequential(use_desi=True, use_h0=False, use_cc=True, use_cmb=True,
                             use_bbn=True, use_sn=None):
    from scipy.optimize import minimize
    if use_sn is None:
        use_sn = Include_SNe

    if Include_SNe:
        x0_red  = [0.022, 0.315, 68.0, -19.3]
        bnd_red = [(0.020, 0.025), (0.10, 0.49), (50.0, 100.0), (-50.0, 50.0)]
    else:
        x0_red  = [0.022, 0.3166, 67.2]
        bnd_red = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0)]

    print("=" * 70)
    print(f"STEP 1: Minimizing with z0 FIXED to {fixed_z0_value}")
    print("=" * 70)

    def chi2_fixed(reduced):
        return chi_squared(_theta_full(reduced, fixed_z0_value), use_desi=use_desi, use_h0=use_h0,
                           use_cc=use_cc, use_cmb=use_cmb, use_bbn=use_bbn, use_sn=use_sn)

    res_fixed = minimize(chi2_fixed, x0_red, method='L-BFGS-B', bounds=bnd_red,
                         options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 2000})
    p = res_fixed.x
    print(f"Fitted (z0={fixed_z0_value} fixed):", np.array2string(p, precision=6))
    chi_squared(_theta_full(p, fixed_z0_value), use_desi=use_desi, use_h0=use_h0, use_cc=use_cc,
                use_cmb=use_cmb, use_bbn=use_bbn, use_sn=use_sn, print_output=True)

    if fix_z0:
        return res_fixed, None

    print("\n" + "=" * 70)
    print("STEP 2: Minimizing with z0 FREE")
    print("=" * 70)

    x0_free = _theta_full(p, 0.0)
    # Bounds in theta layout [w_b, omega_m, H0, z0, (M_B)]
    if Include_SNe:
        bnd_free = [bnd_red[0], bnd_red[1], bnd_red[2], (-0.10, 0.10), bnd_red[3]]
    else:
        bnd_free = [bnd_red[0], bnd_red[1], bnd_red[2], (-0.10, 0.10)]

    def chi2_free(theta):
        return chi_squared(list(theta), use_desi=use_desi, use_h0=use_h0, use_cc=use_cc,
                           use_cmb=use_cmb, use_bbn=use_bbn, use_sn=use_sn)

    res_free = minimize(chi2_free, x0_free, method='L-BFGS-B', bounds=bnd_free,
                        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 2000})
    q = res_free.x
    print("Fitted (z0 free):", np.array2string(q, precision=6))
    chi_squared(list(q), use_desi=use_desi, use_h0=use_h0, use_cc=use_cc,
                use_cmb=use_cmb, use_bbn=use_bbn, use_sn=use_sn, print_output=True)

    print("\n" + "=" * 70)
    print(f"Chi2 (z0=0 fixed): {res_fixed.fun:.3f}")
    print(f"Chi2 (z0 free):    {res_free.fun:.3f}")
    print(f"Delta Chi2:        {res_fixed.fun - res_free.fun:.3f}")
    print(f"Significance:      {np.sqrt(max(0.0, res_fixed.fun - res_free.fun)):.2f} sigma")
    print("=" * 70)
    return res_fixed, res_free

if compute_minimum_chi2 and nested_sampling_data != "desi_only":
    minimize_chi2_sequential(use_desi=True, use_h0=False, use_cc=True, use_cmb=True,
                             use_bbn=True, use_sn=Include_SNe)

# ── Dynesty nested sampling ────────────────────────────────────────────────────
def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

if nested_sampling_data == "desi_only":
    ndim = 2 if fix_z0 else 3
else:
    ndim = 3 + (0 if fix_z0 else 1) + (1 if Include_SNe else 0)

def prior_transform(u):
    if nested_sampling_data == "desi_only":
        omega_m = 0.01 + 0.48 * u[0]
        H0rd    = 5000.0 + 8000.0 * u[1]
        if fix_z0:
            return [omega_m, H0rd]
        z0 = -0.10 + 0.20 * u[2]
        return [omega_m, H0rd, z0]
    w_b     = 0.020 + 0.005 * u[0]    # [0.020, 0.025]
    omega_m = 0.01  + 0.48  * u[1]    # [0.01,  0.49]
    H0      = 50.0  + 50.0  * u[2]    # [50.0,  100.0]
    out = [w_b, omega_m, H0]
    idx = 3
    if not fix_z0:
        out.append(-0.10 + 0.20 * u[idx]); idx += 1   # z0 ∈ [−0.10, 0.10]
    if Include_SNe:
        out.append(-50.0 + 100.0 * u[idx])            # M_B ∈ [−50, 50]
    return out

def results_filename(data=None, fixz0=None):
    if data is None:  data = nested_sampling_data
    if fixz0 is None: fixz0 = fix_z0
    if not fixz0:
        tag = "z0free"
    elif fixed_z0_value == 0.0:
        tag = "LCDM"
    else:
        tag = "z0fix" + str(fixed_z0_value).replace('.', 'p').replace('-', 'm')
    sn   = "_SN" if Include_SNe else ""
    zstr = str(z_min).replace('.', 'p')
    return f"dynesty_results_{tag}{sn}_zmin{zstr}_{data}.npz"

import multiprocessing
from functools import partial
from dynesty import NestedSampler

if perform_nested_sampling:
    _sn = Include_SNe
    # NOTE: partial() of the module-level log_likelihood (not a lambda) so the
    # likelihood is picklable for the multiprocessing Pool workers.
    if nested_sampling_data == "CMB_only":
        likelihood_func = partial(log_likelihood, use_desi=False, use_h0=False,
                                  use_cc=False, use_cmb=True,
                                  use_bbn=False, use_sn=False)
    elif nested_sampling_data == "all":
        likelihood_func = partial(log_likelihood, use_desi=True, use_h0=True,
                                  use_cc=True, use_cmb=True,
                                  use_bbn=True, use_sn=_sn)
    elif nested_sampling_data == "no_desi":
        likelihood_func = partial(log_likelihood, use_desi=False, use_h0=True,
                                  use_cc=True, use_cmb=True,
                                  use_bbn=True, use_sn=_sn)
    elif nested_sampling_data == "no_h0":
        likelihood_func = partial(log_likelihood, use_desi=True, use_h0=False,
                                  use_cc=True, use_cmb=True,
                                  use_bbn=True, use_sn=_sn)
    elif nested_sampling_data == "CMB+bbn+CC":
        likelihood_func = partial(log_likelihood, use_desi=False, use_h0=False,
                                  use_cc=True, use_cmb=True,
                                  use_bbn=True, use_sn=False)
    elif nested_sampling_data == "desi_only":
        likelihood_func = partial(log_likelihood, use_desi=True, use_h0=False,
                                  use_cc=False, use_cmb=False,
                                  use_bbn=False, use_sn=False)
    else:
        likelihood_func = log_likelihood

    print(f"Nested sampling: data={nested_sampling_data}  fix_z0={fix_z0}  "
          f"Include_SNe={Include_SNe}  z_min={z_min}  ndim={ndim}")

    with multiprocessing.Pool(processes=nprocs) as pool:
        sampler = NestedSampler(
            likelihood_func, prior_transform, ndim,
            nlive=500, bound='multi', sample='rwalk',
            pool=pool, queue_size=nprocs)
        sampler.run_nested(dlogz=0.01, print_progress=True)
        results = sampler.results

    logZ    = results.logz[-1]
    logZerr = results.logzerr[-1]
    print(f"Log evidence: {logZ:.3f} ± {logZerr:.3f}")

    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    print("Posterior mean:", np.average(samples, axis=0, weights=weights))

    if nested_sampling_data != "desi_only":
        log_posteriors = np.apply_along_axis(log_posterior, 1, samples)
        _fname = results_filename()
        np.savez(_fname, samples=samples, weights=weights, logp_posteriors=log_posteriors,
                 logZ=logZ, logZerr=logZerr)
    else:
        _fname = results_filename()
        np.savez(_fname, samples=samples, weights=weights, logZ=logZ, logZerr=logZerr)
    print(f"Results saved to {_fname}")

# =============================================================================
# POST-PROCESSING ANALYSIS BLOCKS (default off; run locally on existing chains)
# =============================================================================

def age_of_universe(omega_m, Omega_gamma, Omega_nu, Omega_nu_ur, a_nr_pair, H0):
    def _integ(a):
        z = 1.0 / a - 1.0
        E = a * reduced_hubble_factor(z, omega_m, Omega_gamma, Omega_nu, Omega_nu_ur,
                                      a_nr_pair, Massive_Neutrinos=True)
        return 1.0 / E
    integral, _ = quad(_integ, 0.0, 1.0)
    t0 = integral / H0 * Mpc / 1e3 / Gyr
    return t0

def sound_horizon_drag(w_b, w_bc, z0):
    """Sound horizon at the drag epoch (metres) for the z0 shift."""
    _, r_d, _, _, _, _, _ = _cmb_quantities(w_b, w_bc, z0)
    return r_d

def weighted_percentile(data, weights, percentiles):
    result = np.zeros((len(percentiles), data.shape[1]))
    for i in range(data.shape[1]):
        order = np.argsort(data[:, i])
        sd = data[order, i]; sw = weights[order]
        cum = np.cumsum(sw); cum /= cum[-1]
        for j, p in enumerate(percentiles):
            result[j, i] = np.interp(p / 100.0, cum, sd)
    return result

# ── z0 redshift-binning test ──────────────────────────────────────────────────
def make_filtered_desi_data(z_cut):
    """(z_arr, data_arr, C_inv_arr, sizes) keeping only DESI bins with z > z_cut,
    from the FULL DESI set (independent of the global z_min)."""
    return _filter_desi(z_cut)

def _minimize_binning(z_arr, data_arr, C_inv_arr, sizes):
    """Fit [w_b, omega_m, H0, z0] (z0 free) to a custom DESI subset + CMB/BBN/CC."""
    from scipy.optimize import minimize as sp_minimize

    def chi2_custom(theta):
        w_b, omega_m, H0, z0 = theta
        h = H0 / 100.0
        w_bc = omega_m * h**2
        r_rec, r_d, z_rec, w_gamma, w_nu, w_nu_ur, a_nr_pair = _cmb_quantities(w_b, w_bc, z0)
        Og = w_gamma/h**2; On = w_nu/h**2; Ou = w_nu_ur/h**2
        chi2 = 0.0
        # CMB
        DM_rec = _dc_arcsinh(omega_m, Og, On, Ou, a_nr_pair, h*H_100, z_rec)
        r = np.array([100.0*(r_rec/DM_rec), w_b/(1+z0)**3, w_bc/(1+z0)**3]) - mu_cmb
        chi2 += r @ inv_cov_cmb @ r
        # BBN
        chi2 += ((w_b/(1+z0)**3 - w_b_bbn) / ew_b_bbn)**2
        # DESI subset
        preds = model_predictions(z_arr, omega_m, Og, On, Ou, a_nr_pair,
                                  H0*H0_to_SI, r_d, z0, sizes=sizes)
        diff = preds - data_arr
        chi2 += diff @ C_inv_arr @ diff
        # CC
        z_CC_cosmo = (z_CC - z0) / (1 + z0)
        H_pred = H0 * reduced_hubble_factor(z_CC_cosmo, omega_m, Og, On, Ou, a_nr_pair,
                                            Massive_Neutrinos=True)
        chi2 += np.sum(((H_pred - H_z_CC) / sigma_H_z_CC)**2)
        return chi2 if np.isfinite(chi2) else 1e30

    x0 = [0.0224, 0.30, 67.5, 0.0]
    bounds = [(0.020, 0.025), (0.01, 0.49), (50.0, 100.0), (-0.10, 0.10)]
    res = sp_minimize(chi2_custom, x0, method='L-BFGS-B', bounds=bounds)
    return res.x, res.fun

if compute_z0_redshift_binning:
    print("=" * 70)
    print("z0 REDSHIFT-BINNING TEST (z0 fitted at progressively lower cutoffs)")
    print("=" * 70)
    z_min_cuts = [0.0, 0.295, 0.510, 0.706, 0.934]
    rows = []
    for z_cut in z_min_cuts:
        z_arr_f, data_arr_f, C_inv_f, sizes_f = make_filtered_desi_data(z_cut)
        params, chi2_val = _minimize_binning(z_arr_f, data_arr_f, C_inv_f, sizes_f)
        w_b_f, omega_m_f, H0_f, z0_f = params
        rows.append((z_cut, len(z_arr_f), z0_f, H0_f, chi2_val))
        print(f"  z_min={z_cut:.3f}  N_bins={len(z_arr_f):2d}  "
              f"z0={z0_f:+.5f}  H0={H0_f:.3f}  chi2={chi2_val:.3f}")
    _arr = np.array(rows)
    np.savetxt("z0_redshift_binning.txt", _arr,
               header="z_min  N_bins  z0_best  H0_best  chi2",
               fmt="%.4f  %d  %+.6f  %.4f  %.4f")
    print("  Results saved to z0_redshift_binning.txt")
    print("=" * 70)

# ── z0–H0 posterior correlation ────────────────────────────────────────────────
if compute_z0_H0_correlation:
    print("=" * 70)
    print("z0–H0 POSTERIOR CORRELATION")
    print("=" * 70)
    _chain_file = results_filename(fixz0=False)
    try:
        _data = np.load(_chain_file)
        _samples = _data['samples']; _weights = _data['weights']
        print(f"  Loaded chain: {_chain_file}  ({_samples.shape[0]} samples)")
    except FileNotFoundError:
        print(f"  Chain file not found: {_chain_file} (run with fix_z0=False first)")
        _samples = None
    if _samples is not None and _samples.shape[1] >= 4:
        _H0 = _samples[:, 2]; _z0 = _samples[:, 3]
        _w = _weights / _weights.sum()
        _H0m = np.dot(_w, _H0); _z0m = np.dot(_w, _z0)
        _H0s = np.sqrt(np.dot(_w, (_H0 - _H0m)**2))
        _z0s = np.sqrt(np.dot(_w, (_z0 - _z0m)**2))
        _r = np.dot(_w, (_H0 - _H0m) * (_z0 - _z0m)) / (_H0s * _z0s)
        print(f"  H0 = {_H0m:.3f} ± {_H0s:.3f} km/s/Mpc")
        print(f"  z0 = {_z0m:+.5f} ± {_z0s:.5f}")
        print(f"  Pearson r(z0, H0) = {_r:+.4f}")
        _sig = _z0m / _z0s if _z0s > 0 else 0.0
        print(f"  z0 significance: {_sig:+.2f} sigma")
    print("=" * 70)

# ── BAO significance in data space (from a CMB+bbn+CC chain) ───────────────────
if compute_BAO_significance_data_space:
    print("=" * 70)
    print("BAO SIGNIFICANCE (data space) from CMB+bbn+CC chain")
    print("=" * 70)
    _f = results_filename(data="CMB+bbn+CC")
    try:
        _d = np.load(_f); _s = _d['samples']; _w = _d['weights']
    except FileNotFoundError:
        raise FileNotFoundError(f"Need {_f}: run nested sampling with NESTED_DATA=CMB+bbn+CC first.")
    BAO_pred = np.zeros((_s.shape[0], Data_DESI.shape[0]))
    for i, theta in enumerate(_s):
        w_b, omega_m, H0 = theta[0], theta[1], theta[2]
        z0 = fixed_z0_value if fix_z0 else theta[3]
        h = H0 / 100.0; w_bc = omega_m * h**2
        _, r_d, _, w_gamma, w_nu, w_nu_ur, a_nr_pair = _cmb_quantities(w_b, w_bc, z0)
        BAO_pred[i] = model_predictions(z_DESI, omega_m, w_gamma/h**2, w_nu/h**2,
                                        w_nu_ur/h**2, a_nr_pair, H0*H0_to_SI, r_d, z0)
    mean_pred = np.average(BAO_pred, axis=0, weights=_w)
    cov_pred  = np.cov(BAO_pred.T, aweights=_w)
    diff = Data_DESI - mean_pred
    Cov_DESI = np.linalg.inv(C_inv_total)        # data covariance (un-Cholesky'd)
    inv_tot = np.linalg.inv(Cov_DESI + cov_pred)
    chi2_sig = diff @ inv_tot @ diff
    from scipy.stats import chi2 as _chi2d, norm as _normd
    dof = len(Data_DESI)
    p = _chi2d.sf(chi2_sig, dof)
    print(f"  chi2={chi2_sig:.4f}  dof={dof}  sigma={_normd.isf(p/2):.4f}  p={p:.4e}")
    print("=" * 70)

# ── BAO significance in parameter space (CMB+bbn+CC vs desi_only) ──────────────
if compute_BAO_significance_parameter_space:
    print("=" * 70)
    print("BAO SIGNIFICANCE (parameter space) CMB+bbn+CC vs desi_only")
    print("=" * 70)
    _f_cmb  = results_filename(data="CMB+bbn+CC")
    _f_desi = results_filename(data="desi_only")
    try:
        _dc = np.load(_f_cmb);  _sc = _dc['samples']; _wc = _dc['weights']
        _dd = np.load(_f_desi); _sd = _dd['samples']; _wd = _dd['weights']
    except FileNotFoundError:
        raise FileNotFoundError(f"Need {_f_cmb} and {_f_desi}.")
    omega_m_d = _sd[:, 0]; H0rd_d = _sd[:, 1]
    om_d_mean = np.average(omega_m_d, weights=_wd)
    h0rd_d_mean = np.average(H0rd_d, weights=_wd)
    cov_d = np.cov(np.stack((omega_m_d, H0rd_d)), aweights=_wd)
    z0 = fixed_z0_value
    wb_c = _sc[:, 0]; om_c = _sc[:, 1]; h_c = _sc[:, 2] / 100.0
    wbc_c = om_c * h_c**2
    r_d_c = np.array([sound_horizon_drag(wb_c[i], wbc_c[i], z0) for i in range(len(wb_c))])
    H0rd_c = (100.0 * h_c) * (r_d_c / Mpc)   # H0[km/s/Mpc]·r_d[Mpc]
    _, w_nu_c, _, _, _, _, _ = functions_of_Tcmb(T_CMB_Firas * (1 + z0))
    omega_m_c = om_c + w_nu_c / h_c**2
    om_c_mean = np.average(omega_m_c, weights=_wc)
    h0rd_c_mean = np.average(H0rd_c, weights=_wc)
    cov_c = np.cov(np.stack((omega_m_c, H0rd_c)), aweights=_wc)
    md = np.array([om_d_mean - om_c_mean, h0rd_d_mean - h0rd_c_mean])
    chi2_par = md @ np.linalg.inv(cov_d + cov_c) @ md
    from scipy.stats import chi2 as _chi2p, norm as _normp
    p = _chi2p.sf(chi2_par, 2)
    print(f"  chi2={chi2_par:.4f}  dof=2  sigma={_normp.isf(p/2):.4f}  p={p:.4e}")
    print("=" * 70)

# ── Universe age from a chain ──────────────────────────────────────────────────
if compute_Universe_age:
    _f = results_filename()
    try:
        _d = np.load(_f); _s = _d['samples']; _w = _d['weights']
    except FileNotFoundError:
        raise FileNotFoundError(f"Need {_f}.")
    w_gamma0, w_nu0, w_nu_ur0, a_nr_pair0, _, _, _ = functions_of_Tcmb(T_CMB_Firas)
    ages = np.array([age_of_universe(th[1], w_gamma0/(th[2]/100)**2, w_nu0/(th[2]/100)**2,
                                     w_nu_ur0/(th[2]/100)**2, a_nr_pair0, th[2])
                     for th in _s])
    mean_age = np.average(ages, weights=_w)
    std_age = np.sqrt(np.average((ages - mean_age)**2, weights=_w))
    print(f"Universe age: {mean_age:.3f} ± {std_age:.3f} Gyr")

# ── Cosmological parameter summary from a chain ────────────────────────────────
if compute_cosmological_parameters:
    _f = results_filename()
    try:
        _d = np.load(_f); _s = _d['samples']; _w = _d['weights']
    except FileNotFoundError:
        raise FileNotFoundError(f"Need {_f}.")
    names = ['w_b', 'omega_m', 'H0'] + ([] if fix_z0 else ['z0']) + (['M_B'] if Include_SNe else [])
    means = np.average(_s, axis=0, weights=_w)
    lo = weighted_percentile(_s, _w, [15.87])[0]
    hi = weighted_percentile(_s, _w, [84.13])[0]
    print("Cosmological parameters:")
    for k, nm in enumerate(names):
        print(f"  {nm}: {means[k]:.6f} (+{hi[k]-means[k]:.6f} / -{means[k]-lo[k]:.6f})")
