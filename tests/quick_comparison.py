"""Quick comparison: cosmocnc vs cosmocnc_jax (~2-3 min total).

Compares:
  1. Cosmological background (D_A, E_z, rho_c)
  2. HMF matrix
  3. Cluster abundance (n_tot, dn/dz, abundance_matrix)
  4. Binned likelihood
  5. Unbinned likelihood (data_lik_from_abundance=True)
  6. Backward convolutional likelihood (low-res: n_points_data_lik=64)
  7. Stacked likelihood

Runs on CPU with classy_sz for both packages.
"""

import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = _N_THREADS
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import sys

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc
import cosmocnc_jax

# ── Shared config ──

COSMO = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
    "k_cutoff": 1e8, "ps_cutoff": 1,
}

SCAL_REL = {
    "corr_lnq_lnp": 0.,
    "bias_sz": 0.8,
    "dof": 0.,
}

BASE_CNC = {
    "cluster_catalogue": "SO_sim_0",
    "obs_select": "q_so_sim",
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16,
    "n_points": 16384,
    "sigma_mass_prior": 10,
    "hmf_type": "Tinker08", "hmf_calc": "cnc",
    "mass_definition": "500c",
    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm",
    "interp_tinker": "linear",
    "bins_edges_z": np.linspace(0.01, 3., 9),
    "bins_edges_obs_select": np.exp(np.linspace(np.log(5.), np.log(200.), 7)),
}


def make_instance(pkg, cnc_extra, tool="classy_sz"):
    nc = pkg.cluster_number_counts()
    p = dict(BASE_CNC)
    p["cosmology_tool"] = tool if pkg.__name__ == "cosmocnc" else "classy_sz_jax"
    p.update(cnc_extra)
    nc.cnc_params.update(p)
    nc.cosmo_params.update(COSMO)
    nc.scal_rel_params.update(SCAL_REL)
    nc.initialise()
    return nc


def compare(name, a, b, rtol=1e-4, atol=1e-10):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape {a.shape} vs {b.shape}")
        return False
    max_abs = np.max(np.abs(a - b)) if a.size else 0.
    denom = np.maximum(np.abs(a), 1e-30)
    max_rel = np.max(np.abs(a - b) / denom) if a.size else 0.
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag} {name}: max_rel={max_rel:.3e}, max_abs={max_abs:.3e}")
    return ok


def main():
    t_total = time.time()
    all_ok = True

    # ── Phase 1: HMF + abundance + binned + unbinned-from-abundance ──
    print("=" * 60)
    print("PHASE 1: HMF, abundance, binned & unbinned-from-abundance")
    print("=" * 60)

    cnc_extra = {
        "observables": [["q_so_sim"]],
        "data_lik_from_abundance": True,
        "likelihood_type": "unbinned",
        "n_points_data_lik": 2048,
        "downsample_hmf_bc": 2,
        "delta_m_with_ref": True,
        "scalrel_type_deriv": "numerical",
    }

    print("  Init cosmocnc...")
    t0 = time.time()
    nc_np = make_instance(cosmocnc, cnc_extra)
    print(f"    done ({time.time()-t0:.1f}s)")

    print("  Init cosmocnc_jax...")
    t0 = time.time()
    nc_jx = make_instance(cosmocnc_jax, cnc_extra)
    print(f"    done ({time.time()-t0:.1f}s)")

    # Number counts
    nc_np.get_number_counts()
    nc_jx.get_number_counts()

    # Cosmological background (emulator vs emulator: ~1e-4 expected)
    all_ok &= compare("D_A", nc_np.D_A, nc_jx.D_A, rtol=5e-4)
    all_ok &= compare("E_z", nc_np.E_z, nc_jx.E_z, rtol=5e-4)
    all_ok &= compare("rho_c", nc_np.rho_c, nc_jx.rho_c, rtol=5e-4)

    # HMF
    all_ok &= compare("hmf_matrix", nc_np.hmf_matrix, nc_jx.hmf_matrix, rtol=5e-4)

    # Abundance
    all_ok &= compare("n_tot", nc_np.n_tot, nc_jx.n_tot, rtol=5e-4)
    all_ok &= compare("dn_dz", nc_np.n_z, nc_jx.n_z, rtol=5e-4)
    if nc_np.abundance_matrix is not None:
        all_ok &= compare("abundance_matrix", nc_np.abundance_matrix, nc_jx.abundance_matrix, rtol=1e-3, atol=1e-2)

    # Binned likelihood
    nc_np.cnc_params["likelihood_type"] = "binned"
    nc_jx.cnc_params["likelihood_type"] = "binned"
    ll_bin_np = nc_np.get_log_lik()
    ll_bin_jx = nc_jx.get_log_lik()
    all_ok &= compare("log_lik_binned", ll_bin_np, ll_bin_jx, rtol=1e-4)

    # Unbinned from abundance
    nc_np.cnc_params["likelihood_type"] = "unbinned"
    nc_jx.cnc_params["likelihood_type"] = "unbinned"
    ll_ua_np = nc_np.get_log_lik()
    ll_ua_jx = nc_jx.get_log_lik()
    all_ok &= compare("log_lik_unbinned_from_abund", ll_ua_np, ll_ua_jx, rtol=1e-4)

    # Extreme value & C-statistic
    nc_np.get_log_lik_extreme_value(); nc_np.eval_extreme_value_quantities()
    nc_jx.get_log_lik_extreme_value(); nc_jx.eval_extreme_value_quantities()
    all_ok &= compare("obs_select_max_mean", nc_np.obs_select_max_mean, nc_jx.obs_select_max_mean, rtol=5e-4)

    C_np, Cm_np, Cs_np = nc_np.get_c_statistic()
    C_jx, Cm_jx, Cs_jx = nc_jx.get_c_statistic()
    all_ok &= compare("C_observed", C_np, C_jx, rtol=1e-2)
    all_ok &= compare("C_mean", Cm_np, Cm_jx, rtol=5e-4)

    # ── Phase 2: Backward convolutional (low-res) ──
    print("\n" + "=" * 60)
    print("PHASE 2: Backward convolutional (n_points_data_lik=64)")
    print("=" * 60)

    cnc_bc = {
        "observables": [["q_so_sim"], ["p_so_sim"]],
        "data_lik_from_abundance": False,
        "likelihood_type": "unbinned",
        "n_points_data_lik": 64,
        "downsample_hmf_bc": 8,
        "padding_fraction": 0.,
        "bc_chunk_size": 0,
        "delta_m_with_ref": False,
        "scalrel_type_deriv": "analytical",
        "stacked_likelihood": False,
        "apply_obs_cutoff": False,
        "z_errors": False,
        "sigma_mass_prior": 5.,
    }

    print("  Init cosmocnc (bc)...")
    t0 = time.time()
    nc_np_bc = make_instance(cosmocnc, cnc_bc)
    print(f"    done ({time.time()-t0:.1f}s)")

    print("  Init cosmocnc_jax (bc)...")
    t0 = time.time()
    nc_jx_bc = make_instance(cosmocnc_jax, cnc_bc)
    print(f"    done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    ll_bc_np = nc_np_bc.get_log_lik()
    t_np = time.time() - t0

    t0 = time.time()
    ll_bc_jx = nc_jx_bc.get_log_lik()
    t_jx = time.time() - t0

    all_ok &= compare("log_lik_backward_conv", ll_bc_np, ll_bc_jx, rtol=1e-3)
    print(f"  Timing: cosmocnc={t_np:.2f}s, cosmocnc_jax={t_jx:.2f}s")

    # ── Phase 3: Stacked likelihood ──
    print("\n" + "=" * 60)
    print("PHASE 3: Stacked likelihood")
    print("=" * 60)

    cnc_st = {
        "observables": [["q_so_sim"]],
        "data_lik_from_abundance": False,
        "likelihood_type": "unbinned",
        "stacked_likelihood": True,
        "stacked_data": ["p_so_sim_stacked"],
        "compute_stacked_cov": True,
        "n_points_data_lik": 2048,
        "downsample_hmf_bc": 2,
        "delta_m_with_ref": True,
        "scalrel_type_deriv": "numerical",
    }

    print("  Init cosmocnc (stacked)...")
    t0 = time.time()
    nc_np_st = make_instance(cosmocnc, cnc_st)
    print(f"    done ({time.time()-t0:.1f}s)")

    print("  Init cosmocnc_jax (stacked)...")
    t0 = time.time()
    nc_jx_st = make_instance(cosmocnc_jax, cnc_st)
    print(f"    done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    ll_st_np = nc_np_st.get_log_lik()
    t_np = time.time() - t0

    t0 = time.time()
    ll_st_jx = nc_jx_st.get_log_lik()
    t_jx = time.time() - t0

    all_ok &= compare("log_lik_stacked", ll_st_np, ll_st_jx, rtol=1e-3)
    print(f"  Timing: cosmocnc={t_np:.2f}s, cosmocnc_jax={t_jx:.2f}s")

    # ── Summary ──
    print("\n" + "=" * 60)
    dt = time.time() - t_total
    tag = "ALL PASSED" if all_ok else "SOME FAILED"
    print(f"RESULT: {tag}  (total time: {dt:.0f}s)")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
