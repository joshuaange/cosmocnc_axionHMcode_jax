"""Shared configuration for N-D backward conv test suite."""

import numpy as np

BASE_CNC_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": False,
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16,
    "n_points": 16384, "n_points_data_lik": 128,
    "cosmology_tool": "classy_sz_jax",
    "likelihood_type": "unbinned",
    "data_lik_type": "backward_convolutional",
    "stacked_likelihood": False,
    "apply_obs_cutoff": False,
    "sigma_mass_prior": 5., "z_errors": False,
    "delta_m_with_ref": False, "scalrel_type_deriv": "analytical",
    "downsample_hmf_bc": 8, "padding_fraction": 0.,
    "bc_chunk_size": 0,
    "hmf_type": "Tinker08", "hmf_calc": "cnc",
    "mass_definition": "500c",
}

BASE_SCAL_REL = {
    "bias_sz": 0.8, "bias_cmblens": 0.8,
    "sigma_lnq_szifi": 0.2, "sigma_lnp": 0.2, "corr_lnq_lnp": 0.5,
    "A_szifi": -4.439, "alpha_szifi": 1.617, "a_lens": 1., "dof": 0.,
}

BASE_COSMO = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
    "k_cutoff": 1e8, "ps_cutoff": 1,
}

# Scan parameters
SIGMA8_SCAN = np.linspace(0.790, 0.830, 10)
ALENS_SCAN = np.linspace(0.8, 1.2, 10)
SIGMA8_CONV = np.linspace(0.795, 0.825, 7)

# Resolution sweeps
NPTS_2D = [32, 64]
NPTS_1D = [64, 128, 256, 512]
NPTS_PERF = [64, 128, 256]

# Accuracy: which (mode, npts) combos to test
ACC_COMBOS = [("1d", 128), ("1d", 256), ("1d", 512), ("2d", 32), ("2d", 64)]

N_PERF_EVAL = 10


def make_cnc_params(obs_mode, n_pts_dl=128):
    p = dict(BASE_CNC_PARAMS)
    p["n_points_data_lik"] = n_pts_dl
    if obs_mode == "2d":
        p["observables"] = [["q_so_sim", "p_so_sim"]]
    elif obs_mode == "1d":
        p["observables"] = [["q_so_sim"], ["p_so_sim"]]
    else:
        raise ValueError(f"Unknown obs_mode: {obs_mode}")
    return p
