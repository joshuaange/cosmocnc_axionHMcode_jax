"""Re-fit (A_szifi, alpha_szifi) against tszpower.compute_y0 at the tutorial
cosmology (h=0.6766, sigma_8=0.78, m_nu=0.06).

The fit form matches what survey_sr_planck_sim.py evaluates:
    log10(y0) = A_szifi + 2 log10(Ez) - 0.5 log10(h70)
               + alpha_szifi * log10( (M_phys * h) / B / M_pivot )

i.e., with the M_sun/h convention (M passed to tszpower.compute_y0 is M_phys * h).
We do a 2-parameter least-squares fit at FIXED beta=2, gamma=-0.5 (since these
are theory-fixed in the SR module), and report the resulting (A, alpha).
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_cuda_for_jax = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_for_jax
tf.config.set_visible_devices([], "GPU")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

# Tutorial cosmology
COSMO = dict(h=0.6766, Ob0h2=0.02242, Oc0h2=0.1193, sigma_8=0.78,
             n_s=0.9665, m_nu=0.06, tau_reio=0.0544)
B = 1.41
M_pivot = 3.0 * 0.7e14   # 2.1e14
h = COSMO["h"]

# Get A_s from cosmocnc_jax for the same sigma_8 (the fit needs A_s set in tszpower)
sys.path.insert(0, "/home/lxu/scratch/compute_packages/cosmocnc_jax")
from cosmocnc_jax import cluster_number_counts
from cosmocnc_jax.params import (cnc_params_default, cosmo_params_default,
                                 scaling_relation_params_default)

cnc_params = dict(cnc_params_default)
cnc_params["cosmology_tool"] = "classy_sz_jax"
cnc_params["hmf_calc"] = "cnc"
cnc_params["cosmo_param_density"] = "physical"
cnc_params["M_min"] = 1e14
cnc_params["M_max"] = 1e16
cnc_params["z_min"] = 0.005
cnc_params["z_max"] = 3.0
cnc_params["n_z"] = 50
cnc_params["n_points"] = 50
cnc_params["load_catalogue"] = False
cnc_params["likelihood_type"] = "binned"
cnc_params["binned_lik_type"] = "z_and_obs_select"
cnc_params["data_lik_from_abundance"] = False
cnc_params["bins_edges_z"] = np.linspace(0.005, 3.0, 7)
cnc_params["bins_edges_obs_select"] = np.exp(np.linspace(np.log(5.), np.log(200.), 7))
cnc_params["obs_select"] = "q_planck_sim"
cnc_params["observables"] = [["q_planck_sim"]]
cnc_params["obs_select_min"] = 5.0
cnc_params["obs_select_max"] = 200.0
cnc_params["cosmocnc_verbose"] = "minimal"
cnc_params["survey_sr"] = "/home/lxu/scratch/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_sr_planck_sim.py"
cnc_params["survey_cat"] = "/home/lxu/scratch/compute_packages/cosmocnc_jax/cosmocnc_jax/surveys/survey_cat_planck_sim.py"
cnc_params["tszsbi_noise_dir"] = "/scratch/scratch-lxu/tszsbi/noise_files"
cnc_params["tszsbi_filter_name"] = "immf6"
cnc_params["planck_sim_M_pivot"] = M_pivot

cosmo_params = dict(cosmo_params_default)
cosmo_params.update(COSMO)
nc = cluster_number_counts(cnc_params=cnc_params)
nc.cosmo_params = cosmo_params
nc.scal_rel_params = dict(scaling_relation_params_default)
nc.initialise()
nc.update_params(nc.cosmo_params, nc.scal_rel_params)
A_s = float(nc.cosmo_params["A_s"])
print(f"A_s = {A_s:.6e}  (from sigma_8 = {COSMO['sigma_8']})")

# Load tszpower
sys.path.insert(0, "/scratch/scratch-lxu/tszsbi")
from tszpower import classy_sz as tsz_classy_sz
from tszpower.maskedpower import compute_y0

allpars = dict(tsz_classy_sz.get_all_relevant_params())
allpars.update({
    "h": COSMO["h"], "H0": COSMO["h"]*100.0,
    "omega_b": COSMO["Ob0h2"],
    "omega_cdm": COSMO["Oc0h2"],
    "n_s": COSMO["n_s"],
    "m_ncdm": COSMO["m_nu"],
    "tau_reio": COSMO["tau_reio"],
    "ln10^{10}A_s": float(np.log(A_s * 1e10)),
    "A_s": A_s,
    "sigma8": COSMO["sigma_8"],
    "z_min": 0.005, "z_max": 3.0,
    "M_min": 1e14 * h, "M_max": 1e16 * h,
    # tszpower fit params
    "B": 1.41, "c500": 1.156, "alphaGNFW": 1.062,
    "betaGNFW": 5.4807, "gammaGNFW": 0.3292, "P0GNFW": 8.130,
})
try:
    tsz_classy_sz.set(allpars)
except Exception:
    pass

# Fit grid: physical M, z covering the count-integration range
# Use logspace in M and linspace in z. tszpower's count pipeline uses
# M_input = M_phys * h, so we feed compute_y0 with M_phys * h.
M_grid_phys = np.logspace(np.log10(1e14), np.log10(1e16), 25)  # M_sun (physical)
z_grid      = np.linspace(0.05, 2.5, 20)                        # z

# Compute target: log10(y0_tszp(M_phys*h, z))
def E_LCDM(z, h, omega_b, omega_cdm, m_nu=0.06):
    Om0 = (omega_b + omega_cdm) / h**2
    # neutrino density today
    Onu0 = m_nu / (93.14 * h**2)
    Om0 += Onu0
    OL = 1.0 - Om0
    return np.sqrt(Om0 * (1.0 + z)**3 + OL)


targets = []
features = []   # rows of [1.0, log10((M_phys*h*bias_sz)/M_pivot)]
bias_sz = 1.0 / B
for M_phys in M_grid_phys:
    for z in z_grid:
        y0 = float(compute_y0(M=float(M_phys * h), z=float(z), params_values_dict=allpars))
        if not (y0 > 0 and np.isfinite(y0)):
            continue
        Ez = float(E_LCDM(z, h, COSMO["Ob0h2"], COSMO["Oc0h2"], m_nu=COSMO["m_nu"]))
        h70 = h / 0.7
        # y_tilde = log10(y0) - 2*log10(Ez) + 0.5*log10(h70)
        # = A + alpha * log10( M_phys*h*bias_sz / M_pivot )
        y_tilde = np.log10(y0) - 2.0*np.log10(Ez) + 0.5*np.log10(h70)
        x = np.log10(M_phys * h * bias_sz / M_pivot)
        targets.append(y_tilde)
        features.append([1.0, x])

X = np.asarray(features)
y = np.asarray(targets)

# 2-parameter fit (A, alpha)
coef, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
A_fit, alpha_fit = coef
y_pred = X @ coef
rmse = np.sqrt(np.mean((y - y_pred)**2))
print()
print(f"--- Fixed-cosmology fit (h={h}, sigma_8={COSMO['sigma_8']}, m_nu={COSMO['m_nu']}) ---")
print(f"Fit form: log10(y0) = A + 2*log10(Ez) - 0.5*log10(h70) + alpha*log10((M*h*bias_sz)/M_pivot)")
print(f"  A_szifi (refit)     = {A_fit:.8f}")
print(f"  alpha_szifi (refit) = {alpha_fit:.8f}")
print(f"  RMSE in log10(y0)   = {rmse:.5g} dex  ({(10**rmse - 1)*100:.3f}% in y0)")
print()
print("Compare to old values (test_tszpower.ipynb wide-H0-grid fit):")
print("  A_szifi (old)       = -4.21808934")
print("  alpha_szifi (old)   = 1.12")
print()
print("Implied count change vs current tutorial (1454):")
delta_logy0 = A_fit - (-4.218089337196837)
shift_factor_y0 = 10**delta_logy0
shift_factor_count = shift_factor_y0**2.5
print(f"  delta log10(y0)            = {delta_logy0:+.5f} dex")
print(f"  y0 shift factor            = {shift_factor_y0:.4f}")
print(f"  approx count shift factor (~slope 2.5) = {shift_factor_count:.4f}")
print(f"  predicted new count        = {1454 * shift_factor_count:.1f}")
