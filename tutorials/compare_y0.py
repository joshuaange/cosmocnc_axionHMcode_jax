"""Compare y0(M, z) between cosmocnc_jax's SR fit form and tszpower's
full GNFW compute_y0, on the same (M, z) grid the count integrand uses.
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
import jax.numpy as jnp
import numpy as np

# Cosmology to match tutorial
COSMO = dict(h=0.6766, Ob0h2=0.02242, Oc0h2=0.1193, sigma_8=0.78,
             n_s=0.9665, m_nu=0.06, tau_reio=0.0544)
A_szifi_fit = -4.218089337196837
alpha_szifi = 1.12
B_tszpower = 1.41
M_pivot = 3.0 * 0.7e14  # 2.1e14
bias_sz = 1.0 / B_tszpower

# Load tszpower
sys.path.insert(0, "/scratch/scratch-lxu/tszsbi")
from tszpower import classy_sz as tsz_classy_sz
from tszpower.maskedpower import compute_y0, compute_theta500_arcmin

# Run cosmocnc_jax to get A_s
sys.path.insert(0, "/home/lxu/scratch/compute_packages/cosmocnc_jax")
from cosmocnc_jax import cluster_number_counts
from cosmocnc_jax.params import (cnc_params_default, cosmo_params_default,
                                 scaling_relation_params_default)

# A minimal cosmocnc setup just to get A_s
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
print(f"A_s={A_s:.4e}")

# Configure tszpower's classy_sz
allpars = dict(tsz_classy_sz.get_all_relevant_params())
allpars.update({
    "h": COSMO["h"],
    "omega_b": COSMO["Ob0h2"],
    "omega_cdm": COSMO["Oc0h2"],
    "n_s": COSMO["n_s"],
    "m_ncdm": COSMO["m_nu"],
    "tau_reio": COSMO["tau_reio"],
    "ln10^{10}A_s": float(np.log(A_s * 1e10)),
    "A_s": A_s,
    "sigma8": COSMO["sigma_8"],
    "z_min": 0.005, "z_max": 3.0,
    "M_min": 1e14 * COSMO["h"], "M_max": 1e16 * COSMO["h"],
    # tszpower fit params
    "B": 1.41, "c500": 1.156, "alphaGNFW": 1.062,
    "betaGNFW": 5.4807, "gammaGNFW": 0.3292, "P0GNFW": 8.130,
})
try:
    tsz_classy_sz.set(allpars)
except Exception:
    pass

# Sample (M, z) grid
M_grid = np.geomspace(1e14, 1e16, 5)
z_grid = np.array([0.05, 0.2, 0.5, 1.0, 2.0])

# Helper for cosmocnc-style y0 (the fit form, with M*h fix)
def y0_cnc_form(M_phys, z, A=A_szifi_fit, alpha=alpha_szifi):
    """Match precompute_q_prefactors in survey_sr_planck_sim.py:
       log10(y0) = A + 2 log10(Ez) - 0.5 log10(h70) + alpha*log10(M*h*bias_sz/M_pivot)
    """
    h = COSMO["h"]
    h70 = h / 0.7
    H0c = float(tsz_classy_sz.get_hubble_at_z(0.0, params_values_dict=allpars))
    Hz = float(tsz_classy_sz.get_hubble_at_z(float(z), params_values_dict=allpars))
    Ez = Hz / H0c
    log10_y0 = (A + 2.*np.log10(Ez) - 0.5*np.log10(h70)
                + alpha*np.log10(M_phys * h * bias_sz / M_pivot))
    return 10**log10_y0


# Cosmocnc-style theta_500 (matches survey_sr_planck_sim.py)
const_c = 299792.458   # km/s
def theta500_cnc(M_phys, z):
    h = COSMO["h"]
    H0c = float(tsz_classy_sz.get_hubble_at_z(0.0, params_values_dict=allpars))
    Hz  = float(tsz_classy_sz.get_hubble_at_z(float(z), params_values_dict=allpars))
    Ez = Hz / H0c
    H0 = h * 100.
    DA = float(tsz_classy_sz.get_angular_distance_at_z(float(z), params_values_dict=allpars))
    pref = (6.997 * (H0/70.)**(-2./3.) * (bias_sz/3.)**(1./3.)
            * Ez**(-2./3.) * (500./DA))
    return pref * (M_phys / 1e14)**(1./3.)


h_val = COSMO["h"]
print("\nProbe at the EXACT grid tszpower's count pipeline uses:")
print("  tszpower passes m_grid in [M_min, M_max] = [1e14*h, 1e16*h] (numerical) to compute_y0,")
print("  which means physical M = m_input/h. So for fair compare to cnc (at physical M),")
print("  we feed tszpower compute_y0 with (M_phys * h) to mirror the pipeline.")
print()
print("   M [Msun]   |   z   | y0_tszp (full GNFW, M*h fed) | y0_cnc (fit form) | ratio cnc/tszp | theta500 tszp(M*h) | theta500 cnc | ratio")
print("-"*160)
y0_tszp_arr = np.zeros((len(M_grid), len(z_grid)))
y0_cnc_arr  = np.zeros_like(y0_tszp_arr)
th_tszp_arr = np.zeros_like(y0_tszp_arr)
th_cnc_arr  = np.zeros_like(y0_tszp_arr)
for i, M in enumerate(M_grid):
    for j, z in enumerate(z_grid):
        # Pass M*h to tszpower's functions (this is what its count pipeline does)
        y0t = float(compute_y0(float(M * h_val), float(z), params_values_dict=allpars))
        tht = float(compute_theta500_arcmin(float(M * h_val), float(z), params_values_dict=allpars))
        y0c = y0_cnc_form(M, z)
        thc = theta500_cnc(M, z)
        y0_tszp_arr[i,j] = y0t
        y0_cnc_arr[i,j]  = y0c
        th_tszp_arr[i,j] = tht
        th_cnc_arr[i,j]  = thc
        print(f"  {M:.2e}  | {z:.3f} |  {y0t:.4e}             |   {y0c:.4e}     |    {y0c/y0t:.4f}     |    {tht:6.3f}        |    {thc:6.3f}    | {thc/tht:.4f}")

print("\nSummary: cnc/tszp ratios (across all (M,z))")
ry = y0_cnc_arr / y0_tszp_arr
rt = th_cnc_arr / th_tszp_arr
print(f"  y0 ratio:        min={ry.min():.4f}, max={ry.max():.4f}, median={np.median(ry):.4f}, mean={ry.mean():.4f}, std={ry.std():.4f}")
print(f"  theta500 ratio:  min={rt.min():.4f}, max={rt.max():.4f}, median={np.median(rt):.4f}, mean={rt.mean():.4f}, std={rt.std():.4f}")
print()
print("Implied SNR ratio (cnc/tszp) at a fixed sigma(theta) curve:")
print("  Note: SNR also depends on sigma(theta_500). We need sigma(theta_500_cnc)/sigma(theta_500_tszp)")
print("  to fully answer. But y0_cnc/y0_tszp tells us if our SR amplitude matches.")
