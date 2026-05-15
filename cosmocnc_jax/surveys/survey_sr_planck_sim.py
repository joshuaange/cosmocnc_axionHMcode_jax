"""
Planck-sim survey scaling relations (`survey_sr_planck_sim`).

Single observable: ``q_planck_sim``. Selection σ(θ) is built from
``tszsbi/noise_files`` only (``sigma_dict_szifi.npy`` +
``skyfracs_szifi_cosmology.npy``), as a sky-weighted average over tiles --
identical recipe to ``tszpower.maskedpower.compute_sigma_y0``.

The y0 amplitude is parameterised in the **tszpower-fit form** used in
``tsz_cnc/cnc_number_check/test_tszpower.ipynb``, while preserving
``tszpower``'s ``M_sun / h`` numerical convention in compute_y0:

    log10(y0) = A_szifi + 2 log10(Ez) - 0.5 log10(h70)
                + alpha_szifi * log10( (M h) / B / M_pivot )

with ``B = 1 / bias_sz`` (Planck convention M_obs = (1-b) M_true,
``bias_sz = 1 - b``) and ``M_pivot = 3 * 0.7e14`` by default. Note the
``h = H0/100`` factor inside the alpha-term: ``cosmocnc_jax`` evaluates
the SR on a grid in physical ``M_sun``, while ``tszpower``'s pipeline
feeds compute_y0 a number that is in ``M_sun/h`` (i.e. ``M_phys * h``),
so the same y0 reduction is applied here for consistency. The
``theta_500`` normalisation is unchanged (the same ``h`` factor cancels
against the ``h * 3e14`` denominator in tszpower's theta_500 formula).

This module deliberately drops the CMB-lensing observables (no ``p_so_sim``,
``p_so_sim_stacked``) since this notebook does not use lensing.

Key cnc_params keys:
- ``tszsbi_noise_dir``         (default ``$TSZSBI_NOISE_DIR`` or
                                 ``/scratch/scratch-lxu/tszsbi/noise_files``)
- ``tszsbi_filter_name``       (default ``"immf6"``)
- ``tszsbi_theta_min_arcmin``  (default ``0.5``)
- ``tszsbi_theta_max_arcmin``  (default ``32.0``)
- ``planck_sim_M_pivot``       (default ``3 * 0.7e14``)
- ``planck_sim_abundance_fsky``: ``"full_sky"`` (default, ``skyfracs=[1.0]``),
                                 ``"from_noise_files"`` (``[sum(skyfracs)]``),
                                 or a float in (0, 1].
"""
import os
from pathlib import Path
import logging

import numpy as np
import jax
import jax.numpy as jnp

import cosmocnc_jax
from cosmocnc_jax.utils import simpson


# Default M_pivot used in tszpower test_tszpower.ipynb least-squares fit.
_DEFAULT_M_PIVOT = 3.0 * 0.7e14   # = 2.1e14


# =====================================================================
# Pure JAX scaling relation functions (JIT-compatible, no class state)
# =====================================================================

def sr_q_planck_sim_layer0(x0, prefactor_logy0, prefactor_M_500_to_theta,
                           sigma_sz_poly, alpha_szifi):
    """q_planck_sim layer 0: x0 = ln(M / 1e14 M_sun) -> log(y0/sigma)."""
    log_y0 = x0 * alpha_szifi + prefactor_logy0
    log_theta_500 = jnp.log(prefactor_M_500_to_theta) + x0 / 3.
    log_sigma_sz = jnp.polyval(sigma_sz_poly, log_theta_500)
    x1 = log_y0 - log_sigma_sz
    return x1, log_theta_500


def sr_q_planck_sim_layer0_deriv(log_theta_500, sigma_sz_polyder, alpha_szifi):
    return alpha_szifi - jnp.polyval(sigma_sz_polyder, log_theta_500) / 3.


def sr_q_planck_sim_layer1(x0, _pref_logy0, _pref_theta, dof):
    """q_planck_sim layer 1: log(s/n) -> q = sqrt(exp(x)^2 + dof).

    First 2 args after x0 are prefactors (passed through, unused here).
    With ``dof = 0`` (the default we set below), ``q = y0/sigma`` --
    matches ``tszpower.maskedpower.compute_snr``.
    """
    return jnp.sqrt(jnp.exp(x0) ** 2 + dof)


def sr_q_planck_sim_layer1_deriv(x0, dof):
    exp2 = jnp.exp(2. * x0)
    return exp2 / jnp.sqrt(exp2 + dof)


def precompute_q_prefactors(E_z, H0, D_A, A_szifi, bias_sz, alpha_szifi,
                            M_pivot=_DEFAULT_M_PIVOT):
    """Pre-compute q_planck_sim prefactors at one redshift.

    Implements the **tszpower-fit y0 form** with tszpower's
    ``M_sun / h`` numerical convention. ``cosmocnc_jax`` evaluates the SR
    on a mass grid in physical ``M_sun``, while ``tszpower``'s
    ``compute_y0`` is fed a number that is in ``M_sun / h`` units (so it
    silently treats ``M`` as if it were ``M_phys * h`` when plugging
    into ``(M / B / 2.1e14)^alpha``). To reproduce the same y0(M, z) that
    ``tszpower`` produces in its count pipeline, we multiply M by h here.

    With x0 = ln(M / 1e14) and h = H0 / 100:

        log_y0 = alpha_szifi * x0 + prefactor_logy0
        log10(y0) = A_szifi + 2 log10(Ez) - 0.5 log10(h70)
                    + alpha_szifi * log10( (M h) / B / M_pivot ),   B = 1/bias_sz

    Note: ``theta_500`` already absorbs the same h factor through tszpower's
    ``(M / (h * 3e14))^(1/3)`` form, which is algebraically identical to
    ``(bias_sz/3)^(1/3) * (M_phys/1e14)^(1/3)``, so it is unchanged here.

    ``M_pivot`` defaults to ``3 * 0.7e14``, the value used in
    ``test_tszpower.ipynb`` to fit ``A_szifi``, ``alpha_szifi``.
    """
    h70 = H0 / 70.
    h = H0 / 100.
    prefactor_logy0 = (jnp.log(10.) * A_szifi
                       + 2. * jnp.log(E_z)
                       - 0.5 * jnp.log(h70)
                       + alpha_szifi * jnp.log(bias_sz * h * 1e14 / M_pivot))
    prefactor_M_500_to_theta = (6.997 * (H0 / 70.) ** (-2. / 3.)
                                * (bias_sz / 3.) ** (1. / 3.)
                                * E_z ** (-2. / 3.) * (500. / D_A))
    return prefactor_logy0, prefactor_M_500_to_theta


# =====================================================================
# Class: scaling_relations
# =====================================================================

class scaling_relations:

    def __init__(self, observable="q_planck_sim", cnc_params=None, catalogue=None):
        self.logger = logging.getLogger(__name__)
        self.observable = observable
        self.cnc_params = cnc_params or {}
        self.preprecompute = False
        self.catalogue = catalogue
        self.root_path = cosmocnc_jax.root_path
        self.M_pivot = float(self.cnc_params.get("planck_sim_M_pivot", _DEFAULT_M_PIVOT))

        if observable != "q_planck_sim":
            raise ValueError(
                f"survey_sr_planck_sim only supports observable='q_planck_sim'; "
                f"got '{observable}'. CMB lensing branches were removed because "
                f"the Planck-only tutorial does not use lensing."
            )

    def get_n_layers(self):
        return 2

    def get_n_layers_stacked(self):
        return self.get_n_layers()

    def initialise_scaling_relation(self, cosmology=None):
        self.const = cosmocnc_jax.constants()

        nf = Path(
            self.cnc_params.get(
                "tszsbi_noise_dir",
                os.environ.get("TSZSBI_NOISE_DIR", "/scratch/scratch-lxu/tszsbi/noise_files"),
            )
        ).resolve()
        sigma_obj_file = nf / "sigma_dict_szifi.npy"
        skyfr_file = nf / "skyfracs_szifi_cosmology.npy"
        filter_name = self.cnc_params.get(
            "tszsbi_filter_name", os.environ.get("TSZSBI_FILTER_NAME", "immf6"),
        )
        theta_min_arcmin = float(self.cnc_params.get("tszsbi_theta_min_arcmin", 0.5))
        theta_max_arcmin = float(self.cnc_params.get("tszsbi_theta_max_arcmin", 32.0))

        sigma_obj = np.load(sigma_obj_file, allow_pickle=True).item()
        skyfr = np.load(skyfr_file).ravel()
        data = sigma_obj[filter_name]
        first = next(iter(data.values()))
        ntheta = len(first)
        theta_500_vec = np.exp(
            np.linspace(np.log(theta_min_arcmin), np.log(theta_max_arcmin), ntheta)
        )
        num = np.zeros(ntheta, dtype=float)
        den = 0.0
        for tile, arr in data.items():
            w = skyfr[int(tile)]
            num += w * np.asarray(arr, dtype=float)
            den += w
        sigma_sz_vec = num / den

        self.theta_500_vec = theta_500_vec
        x = np.log(theta_500_vec)
        y = np.log(sigma_sz_vec)
        sigma_sz_poly_np = np.polyfit(x, y, deg=3)
        self.sigma_sz_poly = jnp.asarray(sigma_sz_poly_np)
        self.sigma_sz_polyder = jnp.asarray(np.polyder(sigma_sz_poly_np))

        # Footprint for cluster abundances. σ(θ) is already a sky-weighted average,
        # so the predicted counts scale with the user-chosen ``skyfracs``: full sky
        # by default. Tile weights enter σ(θ), they do NOT shrink the area.
        _fsky = self.cnc_params.get("planck_sim_abundance_fsky", "full_sky")
        if _fsky == "full_sky" or _fsky is True:
            self.skyfracs = [1.0]
        elif _fsky == "from_noise_files":
            self.skyfracs = [float(np.sum(skyfr))]
        else:
            self.skyfracs = [float(_fsky)]

        q_vec = np.linspace(5.0, 10.0, self.cnc_params["n_points"])
        pdf_fd = np.exp(-((q_vec - 3.0) ** 2) / 1.5 ** 2)
        pdf_fd = pdf_fd / simpson(pdf_fd, x=q_vec)
        self.pdf_false_detection = [q_vec, pdf_fd]

    def precompute_scaling_relation(self, params=None, other_params=None, patch_index=0):
        self.params = params

        E_z = other_params["E_z"]
        H0 = other_params["H0"]
        h70 = H0 / 70.
        h = H0 / 100.
        D_A = other_params["D_A"]
        A_szifi = self.params["A_szifi"]
        alpha = self.params["alpha_szifi"]
        bias_sz = self.params["bias_sz"]

        self.prefactor_logy0 = (
            jnp.log(10.) * A_szifi
            + 2. * jnp.log(E_z)
            - 0.5 * jnp.log(h70)
            + alpha * jnp.log(bias_sz * h * 1e14 / self.M_pivot)
        )
        self.prefactor_M_500_to_theta = (
            6.997 * (H0 / 70.) ** (-2. / 3.) * (bias_sz / 3.) ** (1. / 3.)
            * E_z ** (-2. / 3.) * (500. / D_A)
        )

    def eval_scaling_relation(self, x0, layer=0, patch_index=0, other_params=None):
        if layer == 0:
            log_y0 = x0 * self.params["alpha_szifi"] + self.prefactor_logy0
            log_theta_500 = jnp.log(self.prefactor_M_500_to_theta) + x0 / 3.
            self.log_theta_500 = log_theta_500
            log_sigma_sz = jnp.polyval(self.sigma_sz_poly, log_theta_500)
            x1 = log_y0 - log_sigma_sz
        elif layer == 1:
            x1 = jnp.sqrt(jnp.exp(x0) ** 2 + self.params["dof"])
        else:
            raise ValueError(f"Unsupported layer={layer}")

        self.x1 = x1
        return x1

    def eval_derivative_scaling_relation(self, x0, layer=0, patch_index=0,
                                         scalrel_type_deriv="analytical"):
        dx1_dx0 = None
        if scalrel_type_deriv == "analytical":
            if layer == 0:
                dx1_dx0 = self.params["alpha_szifi"] - jnp.polyval(
                    self.sigma_sz_polyder, self.log_theta_500
                ) / 3.
            elif layer == 1:
                dof = self.params["dof"]
                exp = jnp.exp(2. * x0)
                dx1_dx0 = exp / jnp.sqrt(exp + dof)

        if scalrel_type_deriv == "numerical" or dx1_dx0 is None:
            dx1_dx0 = jnp.gradient(self.x1, x0)
        return dx1_dx0

    def eval_scaling_relation_no_precompute(self, x0, layer=0, patch_index=0,
                                            other_params=None, params=None):
        self.params = params
        self.other_params = other_params

        if layer == 0:
            E_z = other_params["E_z"]
            H0 = other_params["H0"]
            h70 = H0 / 70.
            h = H0 / 100.
            D_A = other_params["D_A"]
            A_szifi = self.params["A_szifi"]
            alpha = self.params["alpha_szifi"]
            bias_sz = self.params["bias_sz"]

            prefactor_logy0 = (
                jnp.log(10.) * A_szifi + 2. * jnp.log(E_z)
                - 0.5 * jnp.log(h70)
                + alpha * jnp.log(bias_sz * h * 1e14 / self.M_pivot)
            )
            log_y0 = prefactor_logy0 + x0 * alpha

            prefactor_M_500_to_theta = (
                6.997 * (H0 / 70.) ** (-2. / 3.) * (bias_sz / 3.) ** (1. / 3.)
                * E_z ** (-2. / 3.) * (500. / D_A)
            )
            log_theta_500 = jnp.log(prefactor_M_500_to_theta) + x0 / 3.
            log_sigma_sz = jnp.polyval(self.sigma_sz_poly, log_theta_500)
            x1 = log_y0 - log_sigma_sz
        elif layer == 1:
            x1 = jnp.sqrt(jnp.exp(x0) ** 2 + self.params["dof"])
        else:
            raise ValueError(f"Unsupported layer={layer}")

        return x1

    def get_mean(self, x0, patch_index=0, scatter=None, compute_var=False, other_params=None):
        raise NotImplementedError(
            "survey_sr_planck_sim has no stacked-mean observable (no p_so_sim)."
        )

    def get_cutoff(self, layer=0):
        if layer == 0:
            return -jnp.inf
        elif layer == 1:
            return self.params["q_cutoff"]
        return -jnp.inf

    # =================================================================
    # Factory methods for cnc.py JIT pipeline
    # =================================================================

    def get_layer_fn(self, layer):
        if layer == 0:
            return sr_q_planck_sim_layer0
        elif layer == 1:
            return sr_q_planck_sim_layer1
        raise ValueError(f"No layer fn for layer={layer}")

    def get_layer_returns_aux(self, layer):
        return layer == 0

    def get_layer_deriv_fn(self, layer):
        if layer == 0:
            return sr_q_planck_sim_layer0_deriv
        elif layer == 1:
            return sr_q_planck_sim_layer1_deriv
        return None

    def get_layer_deriv_uses_aux(self, layer):
        return layer == 0

    def get_prefactor_fn(self):
        M_pivot = float(self.M_pivot)

        def fn(E_z, H0, D_A, A_szifi, bias_sz, alpha_szifi):
            return precompute_q_prefactors(
                E_z, H0, D_A, A_szifi, bias_sz, alpha_szifi, M_pivot=M_pivot,
            )

        return fn

    def get_prefactor_vmap_axes(self):
        return (0, None, 0, None, None, None)

    def get_prefactor_args(self, cosmo_quantities, sr_params):
        return (cosmo_quantities["E_z"], cosmo_quantities["H0"],
                cosmo_quantities["D_A"],
                jnp.float64(sr_params["A_szifi"]),
                jnp.float64(sr_params["bias_sz"]),
                jnp.float64(sr_params["alpha_szifi"]))

    def get_n_prefactors(self):
        return 2

    def get_prefactor_fn_unified(self):
        M_pivot = float(self.M_pivot)

        def fn(E_z, D_A, D_l_CMB, rho_c, H0, D_CMB, gamma, z_val,
               A_szifi, bias_sz, alpha_szifi):
            return precompute_q_prefactors(
                E_z, H0, D_A, A_szifi, bias_sz, alpha_szifi, M_pivot=M_pivot,
            )

        return fn

    def get_prefactor_sr_params(self, sr_params):
        return (jnp.float64(sr_params["A_szifi"]),
                jnp.float64(sr_params["bias_sz"]),
                jnp.float64(sr_params["alpha_szifi"]))

    def get_n_prefactor_sr_params(self):
        return 3

    def get_layer_sr_params(self, layer, sr_params):
        if layer == 0:
            return (self.sigma_sz_poly, jnp.float64(sr_params["alpha_szifi"]))
        elif layer == 1:
            return (jnp.float64(sr_params["dof"]),)
        raise ValueError(f"No layer SR params for layer={layer}")

    def get_layer_deriv_sr_params(self, layer, sr_params):
        if layer == 0:
            return (self.sigma_sz_polyder, jnp.float64(sr_params["alpha_szifi"]))
        elif layer == 1:
            return (jnp.float64(sr_params["dof"]),)
        return ()

    def get_scatter_sigma(self, sr_params):
        return float(sr_params.get("sigma_lnq_szifi", 0.))

    def get_mean_fn(self):
        return None

    def get_mean_fn_sr_params(self, sr_params):
        return ()


class scatter:

    def __init__(self, params=None, catalogue=None):
        self.params = params
        self.catalogue = catalogue

    def get_cov(self, observable1=None, observable2=None,
                patch1=0, patch2=0, layer=0, other_params=None):
        if layer == 0:
            if observable1 == "q_planck_sim" and observable2 == "q_planck_sim":
                return self.params["sigma_lnq_szifi"] ** 2
            return 0.
        elif layer == 1:
            if observable1 == "q_planck_sim" and observable2 == "q_planck_sim":
                return 1.
            return 0.
        return 0.
