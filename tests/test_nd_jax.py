"""N-D backward conv tests — cosmocnc_jax evaluations only.

Runs all JAX evaluations and saves results to a pickle file.
"""

import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = _N_THREADS
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import sys
import pickle
import builtins

_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = print

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc_jax
from nd_config import (
    BASE_SCAL_REL, BASE_COSMO, make_cnc_params,
    SIGMA8_SCAN, ALENS_SCAN, SIGMA8_CONV,
    NPTS_2D, NPTS_1D, NPTS_PERF, ACC_COMBOS, N_PERF_EVAL,
)

print(f"JAX backend: {jax.default_backend()}")

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "plots", "nd_results_jax_classy_sz_jax.pkl")
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# ── Instance pool ──

_pool = {}

def get_instance(mode, npts):
    key = (mode, npts)
    if key in _pool:
        return _pool[key]
    print(f"  [init] jax {mode} n_pts_dl={npts}...")
    t0 = time.time()
    nc = cosmocnc_jax.cluster_number_counts()
    nc.cnc_params.update(make_cnc_params(mode, npts))
    nc.cnc_params["cosmology_tool"] = "classy_sz_jax"
    nc.cnc_params["hmf_calc"] = "cnc"
    nc.scal_rel_params.update(dict(BASE_SCAL_REL))
    nc.cosmo_params.update(dict(BASE_COSMO))
    nc.initialise()
    dt = time.time() - t0
    print(f"  [init] ready ({dt:.1f}s)")
    _pool[key] = nc
    return nc


def eval_log_lik(nc, cosmo=None, scal_rel=None):
    if cosmo is not None or scal_rel is not None:
        c = cosmo if cosmo is not None else nc.cosmo_params
        s = scal_rel if scal_rel is not None else nc.scal_rel_params
        nc.update_params(c, s)
    ll = nc.get_log_lik()
    if hasattr(ll, 'block_until_ready'):
        jax.block_until_ready(ll)
    return float(ll)


def warmup(nc, n=2):
    for _ in range(n):
        eval_log_lik(nc)


def scan_param(nc, param_name, param_values, param_dict_key):
    lls = []
    for val in param_values:
        if param_dict_key == "cosmo":
            c = dict(BASE_COSMO); c[param_name] = val
            ll = eval_log_lik(nc, cosmo=c, scal_rel=BASE_SCAL_REL)
        else:
            sr = dict(BASE_SCAL_REL); sr[param_name] = val
            ll = eval_log_lik(nc, cosmo=BASE_COSMO, scal_rel=sr)
        lls.append(ll)
    return lls


def main():
    t_start = time.time()
    results = {}

    # ── Section 1: Parameter scans at npts=96 (2d) / 128 (1d) ──
    print("=" * 60)
    print("SECTION 1: PARAMETER SCANS (jax)")
    print("=" * 60)

    scan_npts = {"2d": 64, "1d": 128}
    scan_results = {}
    for mode in ["2d", "1d"]:
        nc = get_instance(mode, scan_npts[mode])
        warmup(nc)
        label = f"jax_{mode}"
        print(f"  sigma_8 scan: {label}")
        scan_results[(label, "sigma8")] = scan_param(
            nc, "sigma_8", SIGMA8_SCAN, "cosmo")
        print(f"    done: [{scan_results[(label, 'sigma8')][0]:.2f} ... {scan_results[(label, 'sigma8')][-1]:.2f}]")
        print(f"  a_lens scan: {label}")
        scan_results[(label, "alens")] = scan_param(
            nc, "a_lens", ALENS_SCAN, "scal_rel")
        print(f"    done: [{scan_results[(label, 'alens')][0]:.2f} ... {scan_results[(label, 'alens')][-1]:.2f}]")

    results["scan"] = scan_results

    # ── Section 2: Convergence (sigma_8 scan at each npts) ──
    print("\n" + "=" * 60)
    print("SECTION 2: CONVERGENCE (jax)")
    print("=" * 60)

    conv = {}
    for mode, npts_list in [("2d", NPTS_2D), ("1d", NPTS_1D)]:
        for npts in npts_list:
            nc = get_instance(mode, npts)
            warmup(nc)
            key = (mode, npts)
            print(f"  {mode} n_pts_dl={npts}: scanning sigma_8...")
            conv[key] = scan_param(nc, "sigma_8", SIGMA8_CONV, "cosmo")
            print(f"    done: [{conv[key][0]:.2f} ... {conv[key][-1]:.2f}]")

    results["conv"] = conv

    # ── Section 3: 1D vs 2D (reuse npts=96/128) ──
    print("\n" + "=" * 60)
    print("SECTION 3: 1D vs 2D (jax)")
    print("=" * 60)

    nc_2d = get_instance("2d", 64)
    nc_1d = get_instance("1d", 128)
    results["fiducial_2d"] = eval_log_lik(nc_2d)
    results["fiducial_1d"] = eval_log_lik(nc_1d)
    print(f"  fiducial 2D: {results['fiducial_2d']:.4f}")
    print(f"  fiducial 1D: {results['fiducial_1d']:.4f}")

    diff = []
    for val in ALENS_SCAN:
        sr = dict(BASE_SCAL_REL); sr["a_lens"] = val
        l2d = eval_log_lik(nc_2d, cosmo=BASE_COSMO, scal_rel=sr)
        l1d = eval_log_lik(nc_1d, cosmo=BASE_COSMO, scal_rel=sr)
        diff.append(l2d - l1d)
        print(f"    a_lens={val:.2f}: 2D-1D = {l2d - l1d:.3f}")
    results["alens_diff"] = diff

    # Check: 2D with corr=0 should match 1D
    sr_nocorr = dict(BASE_SCAL_REL); sr_nocorr["corr_lnq_lnp"] = 0.
    l2d_nocorr = eval_log_lik(nc_2d, cosmo=BASE_COSMO, scal_rel=sr_nocorr)
    l1d_nocorr = eval_log_lik(nc_1d, cosmo=BASE_COSMO, scal_rel=sr_nocorr)
    results["nocorr_2d"] = l2d_nocorr
    results["nocorr_1d"] = l1d_nocorr
    print(f"  corr=0 check: 2D={l2d_nocorr:.4f}, 1D={l1d_nocorr:.4f}, "
          f"diff={l2d_nocorr - l1d_nocorr:.6f}")

    # ── Section 4: Performance ──
    print("\n" + "=" * 60)
    print("SECTION 4: PERFORMANCE (jax)")
    print("=" * 60)

    perf = {}
    for npts in NPTS_PERF:
        perf[npts] = {}
        for mode in ["1d", "2d"]:
            if mode == "2d" and npts > 64:
                print(f"  n_pts_dl={npts} {mode}: SKIPPED (OOM risk)")
                continue
            nc = get_instance(mode, npts)
            warmup(nc, n=3)
            times = []
            for _ in range(N_PERF_EVAL):
                t0 = time.time()
                eval_log_lik(nc)
                times.append(time.time() - t0)
            avg_ms = np.mean(times) * 1000
            min_ms = np.min(times) * 1000
            std_ms = np.std(times) * 1000
            perf[npts][mode] = {
                "avg_ms": avg_ms, "min_ms": min_ms, "std_ms": std_ms,
                "times_ms": [t * 1000 for t in times],
            }
            print(f"  n_pts_dl={npts} {mode}: avg={avg_ms:.1f}ms, "
                  f"min={min_ms:.1f}ms, std={std_ms:.1f}ms")
    results["perf"] = perf

    # ── Section 5: Accuracy (fiducial at each npts) ──
    print("\n" + "=" * 60)
    print("SECTION 5: ACCURACY (jax)")
    print("=" * 60)

    acc = {}
    for mode, npts in ACC_COMBOS:
        nc = get_instance(mode, npts)
        warmup(nc)
        ll = eval_log_lik(nc)
        acc[(mode, npts)] = ll
        print(f"  {mode} n_pts_dl={npts}: {ll:.4f}")
    results["acc"] = acc

    # ── Save ──
    results["n_inits"] = len(_pool)
    results["total_time"] = time.time() - t_start
    results["backend"] = str(jax.default_backend())

    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"\nDone: {len(_pool)} inits, {results['total_time']/60:.1f} min")
    print(f"Saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
