"""Compare HMF between cosmocnc_jax (hmf_calc='cnc', Tinker08-500c, log interp)
and tszpower's native classy_sz HMF (massfuncs.get_hmf_grid, T08-500c).

We feed both pipelines the SAME cosmology and check:
  - dn/dlnM (Mpc^-3 vs (Mpc/h)^{-3}) on a common (M, z) grid
  - dV/dz/dOmega
  - The total dN/dz/dOmega and N_tot integrated over (M, z), with NO SR cut.

If these two HMFs agree at the few-percent level, the residual ~10%
discrepancy in cluster counts must come from the SR pipeline; if they
disagree by ~10%, the HMF is the suspect.
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# TF first with no GPU, then JAX with GPU (same trick as the notebook)
_cuda_for_jax = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: F401
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_for_jax
tf.config.set_visible_devices([], "GPU")
try:
    tf.config.experimental.set_visible_devices([], "GPU")
except Exception:
    pass

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
print("JAX devices:", jax.devices())
print("TF GPUs (should be []):", tf.config.get_visible_devices("GPU"))

# --- common cosmology (matches the tutorial / tsz_cnc gencat) ---
COSMO = dict(h=0.6766, Ob0h2=0.02242, Oc0h2=0.1193, sigma_8=0.78,
             n_s=0.9665, m_nu=0.06, tau_reio=0.0544)
Z_MIN, Z_MAX, N_Z = 0.005, 3.0, 200
M_MIN_PHYS, M_MAX_PHYS = 1e14, 1e16  # M_sun (physical)
N_M = 100

print("=" * 70)
print("HMF comparison: cosmocnc_jax (cnc, Tinker08-500c, interp_log) vs")
print("                tszpower native (classy_sz P(k) + Tinker08-500c)")
print("=" * 70)
print(f"Cosmo: {COSMO}")
print(f"z in [{Z_MIN}, {Z_MAX}], n_z={N_Z};  M in [{M_MIN_PHYS:.1e}, "
      f"{M_MAX_PHYS:.1e}] M_sun, n_M={N_M}")
print()

# =====================================================================
# 1) cosmocnc_jax HMF
# =====================================================================
print("--- cosmocnc_jax (hmf_calc='cnc', interp_tinker='log') ---")
sys.path.insert(0, "/home/lxu/scratch/compute_packages/cosmocnc_jax")
from cosmocnc_jax import cluster_number_counts
from cosmocnc_jax.params import (cnc_params_default,
                                 cosmo_params_default,
                                 scaling_relation_params_default)

cnc_params = dict(cnc_params_default)
cnc_params["cosmology_tool"] = "classy_sz_jax"
cnc_params["hmf_calc"] = "cnc"
cnc_params["hmf_type"] = "Tinker08"
cnc_params["mass_definition"] = "500c"
cnc_params["interp_tinker"] = "log"
cnc_params["cosmo_param_density"] = "physical"
cnc_params["M_min"] = M_MIN_PHYS
cnc_params["M_max"] = M_MAX_PHYS
cnc_params["z_min"] = Z_MIN
cnc_params["z_max"] = Z_MAX
cnc_params["n_z"] = N_Z
cnc_params["n_points"] = N_M
cnc_params["load_catalogue"] = False
cnc_params["likelihood_type"] = "binned"
cnc_params["binned_lik_type"] = "z_and_obs_select"
cnc_params["data_lik_from_abundance"] = False
cnc_params["bins_edges_z"] = np.linspace(Z_MIN, Z_MAX, 7)
cnc_params["bins_edges_obs_select"] = np.exp(np.linspace(np.log(5.), np.log(200.), 7))
cnc_params["obs_select"] = "q_planck_sim"
cnc_params["observables"] = [["q_planck_sim"]]
cnc_params["obs_select_min"] = 5.0
cnc_params["obs_select_max"] = 200.0
cnc_params["cosmocnc_verbose"] = "minimal"

# point at our planck_sim survey modules (we won't actually run abundance — just HMF)
PKG_ROOT = "/home/lxu/scratch/compute_packages/cosmocnc_jax/cosmocnc_jax"
cnc_params["survey_sr"] = f"{PKG_ROOT}/surveys/survey_sr_planck_sim.py"
cnc_params["survey_cat"] = f"{PKG_ROOT}/surveys/survey_cat_planck_sim.py"
cnc_params["tszsbi_noise_dir"] = "/scratch/scratch-lxu/tszsbi/noise_files"
cnc_params["tszsbi_filter_name"] = "immf6"

cosmo_params = dict(cosmo_params_default)
for k, v in COSMO.items():
    cosmo_params[k] = v
scal = dict(scaling_relation_params_default)

nc = cluster_number_counts(cnc_params=cnc_params)
nc.cosmo_params = cosmo_params
nc.scal_rel_params = scal
nc.initialise()
nc.update_params(nc.cosmo_params, nc.scal_rel_params)
nc.get_hmf()

# cosmocnc_jax: ln_M = ln(M / 1e14) with M in M_sun (physical)
M_cnc = np.exp(np.asarray(nc.ln_M)) * 1e14            # M_sun
z_cnc = np.asarray(nc.redshift_vec)
hmf_cnc = np.asarray(nc.hmf_matrix)                   # dn/dlnM/(dV) Mpc^-3
# cnc.py multiplies by volume_element internally, so hmf_matrix == dn/dlnM * dV/(dz dOmega)
# i.e. hmf_matrix has units (Mpc/M_sun_step)^... actually let's just check by checking the per-cluster integrand.
# In cnc.py: hmf_row = hmf_matrix[iz], used as dn/dx0 where x0 = ln(M/1e14), and integrated as
#   simpson(hmf_row, x=ln_M, axis=1)
# That gives dN/(dz dOmega). So hmf_matrix is dn/dlnM * dV/(dz dOmega) [in (Mpc^3) * (Mpc^-3) / sr / M_sun_step? no —
# more precisely it has units of N / (sr * dz * dlnM)]

# Strip the volume to compare with tszpower's dn/dlnM
# We can re-run get_hmf with volume_element=False to get pure dn/dlnM
nc.get_hmf(volume_element=False)
M_cnc2 = np.exp(np.asarray(nc.ln_M)) * 1e14
hmf_cnc_pure = np.asarray(nc.hmf_matrix)              # dn/dlnM (Mpc^-3)
print(f"  M grid: [{M_cnc[0]:.3e}, {M_cnc[-1]:.3e}] M_sun, n_M={M_cnc.size}")
print(f"  z grid: [{z_cnc[0]:.4f}, {z_cnc[-1]:.4f}], n_z={z_cnc.size}")
print(f"  dn/dlnM at (M=3e14, z=0.1)  = {np.interp(np.log(3e14), np.log(M_cnc), hmf_cnc_pure[np.argmin(np.abs(z_cnc-0.1))]):.4e}  Mpc^-3")
print(f"  dn/dlnM at (M=1e15, z=0.5)  = {np.interp(np.log(1e15), np.log(M_cnc), hmf_cnc_pure[np.argmin(np.abs(z_cnc-0.5))]):.4e}  Mpc^-3")

# dV/dz/dOmega from cosmocnc_jax (Mpc^3 / sr) -- the volume_element_vec in cnc.py is chi^2/H_over_c
from cosmocnc_jax.emulators import build_cosmo_vec
cosmo = nc.cosmology
cosmo_vec_h = build_cosmo_vec(cosmo._pvd, cosmo._emu_param_orders['h'])
z_with_0 = jnp.concatenate([jnp.array([0.]), nc.redshift_vec])
H_over_c_all = cosmo._predict_H(cosmo_vec_h, z_with_0)
H_over_c_z = H_over_c_all[1:]
cosmo_vec_da = build_cosmo_vec(cosmo._pvd, cosmo._emu_param_orders['da'])
DA = cosmo._predict_DA(cosmo_vec_da, nc.redshift_vec)
chi = DA * (1.0 + nc.redshift_vec)
dV_cnc = np.asarray(chi**2 / H_over_c_z)             # Mpc^3 / sr
print(f"  dV/dzdΩ at z=0.5 = {dV_cnc[np.argmin(np.abs(z_cnc-0.5))]:.4e}  Mpc^3/sr")

# Total N over 4π sr (no SR cut), integrated dn/dlnM dlnM dV/dz dz
from cosmocnc_jax.utils import simpson as cnc_simpson
dndz_cnc = np.asarray(cnc_simpson(jnp.asarray(hmf_cnc_pure) * dV_cnc[:, None] * 4*jnp.pi,
                                  x=jnp.log(M_cnc), axis=1))
N_cnc = float(cnc_simpson(jnp.asarray(dndz_cnc), x=nc.redshift_vec))
print(f"  N_total (no SR cut, 4π sr) = {N_cnc:.4e}")
print()

# =====================================================================
# 2) tszpower native HMF
# =====================================================================
print("--- tszpower native (classy_sz P(k) + mcfit + Tinker08-500c) ---")
sys.path.insert(0, "/scratch/scratch-lxu/tszsbi")
import tszpower
from tszpower import classy_sz as tsz_classy_sz
from tszpower.massfuncs import MF_T08, _tophat_instance

# Configure classy_sz with the same cosmology
A_s = float(nc.cosmo_params["A_s"])
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
    "z_min": Z_MIN,
    "z_max": Z_MAX,
    "M_min": M_MIN_PHYS * COSMO["h"],   # M_sun/h
    "M_max": M_MAX_PHYS * COSMO["h"],
})
# tszpower keeps its own classy_sz instance; let it know we've changed cosmology
try:
    tsz_classy_sz.set(allpars)
except Exception:
    pass

# Reproduce tszpower.massfuncs.get_hmf_grid logic without vmap (TF emulator
# requires concrete z values, so we loop in Python and stack).
rparams = tsz_classy_sz.get_all_relevant_params(params_values_dict=allpars)
h = float(rparams["h"])
print(f"  rparams: h={h}, Omega0_cb={rparams['Omega0_cb']:.5f}, Omega0_m={rparams['Omega0_m']:.5f}")

z_grid_tp = np.asarray(tsz_classy_sz.z_grid())
print(f"  z_grid_tp: [{z_grid_tp[0]:.5f}, {z_grid_tp[-1]:.5f}], n_z={z_grid_tp.size}")

# P(k) at each z (loop, no vmap)
pk_list = []
ks = None
for zi in z_grid_tp:
    pks, _ks = tsz_classy_sz.get_pkl_at_z(float(zi), params_values_dict=allpars)
    pk_list.append(np.asarray(pks).flatten())
    if ks is None:
        ks = np.asarray(_ks).flatten()
P = np.stack(pk_list, axis=0).T   # (n_k, n_z)
print(f"  k range: [{ks[0]:.3e}, {ks[-1]:.3e}], n_k={ks.size}")

# sigma(R), dsigma^2/dR via mcfit TophatVar (numpy backend OK)
from mcfit import TophatVar
tv0 = TophatVar(ks, lowring=True, backend="jax")
var_list, dvar_list = [], []
for iz in range(P.shape[1]):
    rvar, var_z = tv0(P[:, iz], extrap=True)
    var_list.append(np.asarray(var_z))
    dvar_list.append(np.gradient(np.asarray(var_z), np.asarray(rvar)))
var = np.stack(var_list, axis=0)        # (n_z, n_R)
dvar = np.stack(dvar_list, axis=0)
R = np.asarray(rvar)
Rh = R * h
lnm_tp_h = np.log(4*np.pi*rparams["Omega0_cb"]*rparams["Rho_crit_0"]*Rh**3/3.)
sigmas = np.sqrt(var)
dsigma_dR = dvar / (2. * sigmas)

# delta_mean = delta_crit * Omega_m_z (matches classy_sz get_delta_mean_from_delta_crit_at_z)
# classy_sz only handles scalar z; loop in Python.
delta_mean = np.array([
    float(tsz_classy_sz.get_delta_mean_from_delta_crit_at_z(
        500, float(zi), params_values_dict=allpars))
    for zi in z_grid_tp
])
print(f"  delta_mean(z=0)={delta_mean[0]:.2f}, (z=1)={delta_mean[len(z_grid_tp)//2]:.2f}")

hmf_T08 = np.asarray(MF_T08(jnp.asarray(sigmas), jnp.asarray(z_grid_tp), jnp.asarray(delta_mean)))
# dn/dlnM = -f * R * dsigma/dR / (4π σ Rh³)
dndlnm_tp_h = -hmf_T08 * R[None, :] * dsigma_dR / (4*np.pi * sigmas * Rh[None, :]**3)
# Note: lnm_tp_h is just R-dependent (same for all z)
lnx_tp = np.log(1.0 + z_grid_tp)
M_tp_h = np.exp(lnm_tp_h)
M_tp_phys = M_tp_h / h
dndlnm_tp_phys = dndlnm_tp_h * h**3   # convert (Mpc/h)^-3 -> Mpc^-3
z_tp = z_grid_tp
print(f"  M grid (M_sun/h) -> M_phys: [{M_tp_phys[0]:.3e}, {M_tp_phys[-1]:.3e}] M_sun, n_M={M_tp_phys.size}")

# Compute tszpower's volume element via classy_sz (chi^2/H), at our z_cnc grid for direct compare
try:
    chi_z_h = np.array([
        float(tsz_classy_sz.get_chi_at_z(float(zi), params_values_dict=allpars))
        for zi in z_cnc
    ])
    H_z = np.array([
        float(tsz_classy_sz.get_Hubble_at_z(float(zi), params_values_dict=allpars))
        for zi in z_cnc
    ])
    dV_tp_h = chi_z_h**2 / H_z          # (Mpc/h)^3 / sr  (chi in Mpc/h, H in h/Mpc)
    dV_tp_phys = dV_tp_h / h**3          # Mpc^3 / sr
    print(f"  dV/dzdΩ at z=0.5 (Mpc^3/sr): {dV_tp_phys[np.argmin(np.abs(z_cnc-0.5))]:.4e}")
except Exception as e:
    print(f"  [could not compute tszp dV: {e}]")
    dV_tp_phys = dV_cnc.copy()

# Interpolate tszp HMF onto the cosmocnc (z_cnc, M_cnc) grid for a fair comparison
from scipy.interpolate import RegularGridInterpolator
interp_tp = RegularGridInterpolator(
    (lnx_tp, lnm_tp_h), np.log(dndlnm_tp_phys),
    bounds_error=False, fill_value=None)
LNX_q = np.log(1.0 + z_cnc)
LNM_q = np.log(M_cnc * COSMO["h"])  # M in M_sun/h to match lnm_tp_h
pts = np.stack(np.meshgrid(LNX_q, LNM_q, indexing="ij"), axis=-1)
hmf_tp_on_cnc = np.exp(interp_tp(pts))   # dn/dlnM at (z_cnc, M_cnc) in Mpc^-3

# Total N
dndz_tp = np.trapz(hmf_tp_on_cnc * dV_tp_phys[:, None] * 4*np.pi,
                   x=np.log(M_cnc), axis=1)
N_tp = float(np.trapz(dndz_tp, x=z_cnc))
print(f"  dn/dlnM at (M=3e14, z=0.1)  = {np.interp(np.log(3e14*COSMO['h']), lnm_tp_h, dndlnm_tp_phys[np.argmin(np.abs(z_tp-0.1))]):.4e}  Mpc^-3")
print(f"  dn/dlnM at (M=1e15, z=0.5)  = {np.interp(np.log(1e15*COSMO['h']), lnm_tp_h, dndlnm_tp_phys[np.argmin(np.abs(z_tp-0.5))]):.4e}  Mpc^-3")
print(f"  N_total (no SR cut, 4π sr) = {N_tp:.4e}")
print()

# =====================================================================
# 3) Direct ratio cnc / tszp on common (z_cnc, M_cnc) grid
# =====================================================================
print("--- Ratio cosmocnc_jax / tszpower (on common z_cnc, M_cnc grid) ---")
ratio_hmf = hmf_cnc_pure / hmf_tp_on_cnc
print(f"  HMF ratio:  min={ratio_hmf.min():.4f}, max={ratio_hmf.max():.4f}, "
      f"median={np.median(ratio_hmf):.4f}, mean={ratio_hmf.mean():.4f}")
ratio_dV = dV_cnc / dV_tp_phys
print(f"  dV ratio :  min={ratio_dV.min():.4f}, max={ratio_dV.max():.4f}, "
      f"median={np.median(ratio_dV):.4f}")
print(f"  N_total ratio cnc / tszp = {N_cnc / N_tp:.4f}")
print()

# Save grids for inspection
out = {
    "M_phys": M_cnc.tolist(),
    "z": z_cnc.tolist(),
    "hmf_cnc": hmf_cnc_pure.tolist(),
    "hmf_tszp": hmf_tp_on_cnc.tolist(),
    "dV_cnc": dV_cnc.tolist(),
    "dV_tszp": dV_tp_phys.tolist(),
    "N_cnc_total": N_cnc,
    "N_tszp_total": N_tp,
    "cosmo": COSMO,
}
with open("/home/lxu/scratch/compute_packages/cosmocnc_jax/tutorials/compare_hmf_results.json", "w") as f:
    json.dump(out, f)
print("Saved compare_hmf_results.json")
