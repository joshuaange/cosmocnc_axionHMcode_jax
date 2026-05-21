import jax
import jax.numpy as jnp
import jaxopt
import functools
import numpy as np
import time
import logging
from scipy.special import gamma as Gamma
from scipy import integrate
from scipy.optimize import brentq, curve_fit


# =====================================================================
# Tinker08 parameter arrays (module-level constants for JIT)
# =====================================================================

TINKER08_DELTA_LOG = jnp.log10(jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.], dtype=jnp.float64))
TINKER08_DELTA_LIN = jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.], dtype=jnp.float64)
TINKER08_A = jnp.array([0.186,0.2,0.212,0.218,0.248,0.255,0.260,0.260,0.260], dtype=jnp.float64)
TINKER08_a = jnp.array([1.47,1.52,1.56,1.61,1.87,2.13,2.30,2.53,2.66], dtype=jnp.float64)
TINKER08_b = jnp.array([2.57,2.25,2.05,1.87,1.59,1.51,1.46,1.44,1.41], dtype=jnp.float64)
TINKER08_c = jnp.array([1.19,1.27,1.34,1.45,1.58,1.80,1.97,2.24,2.44], dtype=jnp.float64)


# =====================================================================
# Pure JAX functions for JIT compilation
# =====================================================================

def f_sigma_jit(sigma, redshift, Delta, tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                interp_log=True):
    """Pure JAX Tinker08 multiplicity function for JIT compilation.

    All arguments must be JAX arrays/scalars. No Python objects.
    interp_log: bool (static). If True, interpolate in log10(Delta) space (default).
                If False, interpolate in linear Delta space.
    """
    alpha = 10.**(-(0.75/jnp.log10(Delta/75.))**1.2)

    Delta_interp = jnp.where(interp_log, jnp.log10(Delta), Delta)

    A = jnp.interp(Delta_interp, tinker_Delta, tinker_A) * (1.+redshift)**(-0.14)
    a = jnp.interp(Delta_interp, tinker_Delta, tinker_a) * (1.+redshift)**(-0.06)
    b = jnp.interp(Delta_interp, tinker_Delta, tinker_b) * (1.+redshift)**(-alpha)
    c = jnp.interp(Delta_interp, tinker_Delta, tinker_c)

    return A*((sigma/b)**(-a)+1.)*jnp.exp(-c/sigma**2)


def get_sigma_M_from_arrays(M_vec, rho_m, R_vec, sigma_vec, dsigma_vec):
    """Pure JAX function to interpolate sigma(M) and dsigma/dR(M) from precomputed arrays."""
    R = (3. * M_vec / (4. * jnp.pi * rho_m))**(1./3.)
    sigma = jnp.interp(R, R_vec, sigma_vec)
    dsigmadR = jnp.interp(R, R_vec, dsigma_vec)
    return sigma, dsigmadR, R


def compute_hmf_single_z(sigma, dsigmadR, R, M_vec, rho_m, redshift, Delta,
                          tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                          volume_element_val, interp_log=True):
    """Pure JAX: compute HMF for a single redshift from precomputed sigma arrays.

    Returns hmf in log-mass units (dn/dlnM * dV/dz * 4pi).
    volume_element_val: dV/dz/dOmega value for this redshift (already computed outside JIT).
    interp_log: bool (static). If True, interpolate Tinker08 params in log10(Delta).
    """
    dMdR = 4.*jnp.pi*rho_m*R**2

    fsigma = f_sigma_jit(sigma, redshift, Delta, tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                          interp_log=interp_log)

    hmf = -fsigma*rho_m/M_vec/dMdR*dsigmadR/sigma
    M_eval = M_vec/1e14
    hmf = hmf*1e14

    # log=True: multiply by M_eval (= M/1e14) for dn/dlnM
    hmf = hmf*M_eval

    # Apply volume element
    hmf = hmf * volume_element_val

    return hmf


# Vectorized version over redshift dimension
# interp_log is not vmapped (None) — it's a scalar bool shared across redshifts
_compute_hmf_vmap_log = jax.vmap(
    functools.partial(compute_hmf_single_z, interp_log=True),
    in_axes=(0, 0, 0, None, None, 0, 0,
             None, None, None, None, None, 0))

_compute_hmf_vmap_lin = jax.vmap(
    functools.partial(compute_hmf_single_z, interp_log=False),
    in_axes=(0, 0, 0, None, None, 0, 0,
             None, None, None, None, None, 0))


@functools.partial(jax.jit, static_argnums=(14,))
def compute_hmf_matrix_jit(sigma_matrix, dsigma_matrix, R_matrix, M_vec, rho_m,
                            redshift_vec, Delta_vec, volume_element_vec,
                            tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                            M_min_cutoff, interp_log=True):
    """JIT-compiled: compute full HMF matrix from precomputed sigma arrays.

    Args:
        sigma_matrix: (n_z, n_points) sigma values
        dsigma_matrix: (n_z, n_points) dsigma/dR values
        R_matrix: (n_z, n_points) R values
        M_vec: (n_points,) mass vector
        rho_m: scalar mean matter density
        redshift_vec: (n_z,) redshift values
        Delta_vec: (n_z,) overdensity values (w.r.t. mean) per redshift
        volume_element_vec: (n_z,) dV/dz/dOmega values
        tinker_*: Tinker08 parameter arrays
        M_min_cutoff: minimum mass cutoff (or -1 for no cutoff)
        interp_log: bool (static). If True, use log10(Delta) interpolation.

    Returns:
        hmf_matrix: (n_z, n_points)
    """
    if interp_log:
        hmf_matrix = _compute_hmf_vmap_log(sigma_matrix, dsigma_matrix, R_matrix,
                                            M_vec, rho_m, redshift_vec, Delta_vec,
                                            tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                                            volume_element_vec)
    else:
        hmf_matrix = _compute_hmf_vmap_lin(sigma_matrix, dsigma_matrix, R_matrix,
                                            M_vec, rho_m, redshift_vec, Delta_vec,
                                            tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                                            volume_element_vec)

    # Apply M_min_cutoff if needed
    cutoff_mask = jnp.where(M_vec < M_min_cutoff, 0., 1.)
    hmf_matrix = hmf_matrix * cutoff_mask[jnp.newaxis, :]

    return hmf_matrix


class halo_mass_function:

    def __init__(self,
                 cosmology=None,
                 hmf_type="Tinker08",
                 mass_definition="500c",
                 M_min=1e13,M_max=1e16,
                 M_min_cutoff=None,
                 n_points=1000,
                 type_deriv="numerical",
                 hmf_calc="cnc",
                 extra_params=None,
                 logger = None,
                 interp_tinker=None):

        self.hmf_type = hmf_type

        self.mass_definition = mass_definition
        self.cosmology = cosmology
        self.h = self.cosmology.background_cosmology.H0.value/100.

        self.M_min = M_min
        self.M_max = M_max
        self.M_min_cutoff = M_min_cutoff
        self.n_points = n_points
        self.type_deriv = type_deriv
        self.hmf_calc = hmf_calc
        self.extra_params = extra_params

        self.other_params = {"interp_tinker":interp_tinker}

        self.logger = logging.getLogger(__name__)

        self.sigma_r_dict = {}

        self.const = constants()

        if self.hmf_type in ["Tinker08", "ST_axionHMcode"]:

            self.rho_c_0 = self.cosmology.background_cosmology.critical_density(0.).value*self.const.mpc**3/self.const.solar*1e3

        if self.hmf_calc == "hmf":

            import hmf as hmf_package

            if self.mass_definition[-1] == "c":

                md = "SOCritical"

            elif self.mass_definition[-1] == "m":

                md = "SOMean"

            self.massfunc_hmf = hmf_package.MassFunction(Mmax=np.log10(self.M_max*self.h),
                                                         Mmin=np.log10(self.M_min*self.h),
                                                         z=0.,
                                                         mdef_model=md,
                                                         mdef_params={"overdensity":float(self.mass_definition[0:-1])},
                                                         cosmo_model=self.cosmology.background_cosmology,
                                                         dlog10m=0.005,
                                                         sigma_8=cosmology.cosmo_params["sigma_8"],
                                                         n=cosmology.cosmo_params["n_s"])

    def eval_hmf(self,redshift,log=False,volume_element=False,save_sigma_r=False,load_sigma_r=False,
    M_min=None,M_max=None,n_points=None,return_profile_params=False):

        if M_min is None:

            M_min = self.M_min

        if M_max is None:

            M_max = self.M_max

        if n_points is None:

            n_points = self.n_points

        if log == False:

            M_vec = jnp.linspace(M_min,M_max,n_points)

        elif log == True:

            M_vec = jnp.exp(jnp.linspace(jnp.log(M_min),jnp.log(M_max),n_points))

        if self.hmf_calc == "cnc":

            if self.hmf_type == "ST_axionHMcode":

                if self.mass_definition == "200c":
                    Del = 200
                elif self.mass_definition == "500c":
                    Del = 500
                else:
                    print("ST_axionHMcode not yet updated to work with non-200c mass definitions")
                # note: axionHMcode uses h units

                E_z = lambda z: self.cosmology.background_cosmology.H(z).value / (100. * self.h)

                # 200c
                Om0 = self.cosmology.cosmo_params["Om0"]
                rho_crit_z = self.rho_c_0 * E_z(redshift)**2 # Msol/Mpc^3
                R_200c = (3. * M_vec / (4. * np.pi * rho_crit_z* Del))**(1./3.) # Mpc
                # virial
                rho_m = self.rho_c_0 * Om0 # Msol/Mpc^3
                #G_a = func_axionHMcode_D_z_unnorm_int(0., Om0, E_z)_
                #g_a = func_axionHMcode_D_z_unnorm(redshift, Om0, E_z)*(1+redshift)
                g_a = np.interp(redshift, self.cosmology.D_grid_z_full, self.cosmology.D_grid_full) * self.cosmology.normalisation_cached * (1 + redshift)
                G_a = self.cosmology.G_a_cached
                Delta_vir = func_axionHMcode_Delta_vir(redshift, Om0, G_a, E_z, g_a) # note: wrt mean at z=0

                c_min = 5.196 # should turn this into a parameter
                k, ps = self.cosmology.power_spectrum.get_linear_power_spectrum(redshift)
                sigma_r = sigma_R((k, ps), cosmology=self.cosmology)
                sigma_r.get_derivative(type_deriv=self.type_deriv)
                #normalisation = func_axionHMcode_D_z_unnorm(0., Om0, E_z)
                delta_c   = func_axionHMcode_delta_c(redshift, Om0, G_a, E_z, g_a)
                # Solve on coarse grid
                if "n_mass_points_coarse" in self.cosmology.cosmo_params:
                    n_coarse = self.cosmology.cosmo_params["n_mass_points_coarse"]
                    M_vec_coarse = np.exp(np.linspace(np.log(M_vec.min()), np.log(M_vec.max()), n_coarse))
                    R_200c_coarse = (3. * M_vec_coarse / (4. * np.pi * rho_crit_z * Del))**(1./3.)

                    if return_profile_params:
                        Mvir_coarse, R_vir_vec_coarse, r_s_vec_coarse, delta_char_vec_coarse = find_M_vir_from_M_200c(M_vec_coarse, R_200c_coarse, 
                                                           rho_m, rho_crit_z,
                                                           Delta_vir, c_min, redshift, Om0, sigma_r,
                                                           self.cosmology.normalisation_cached, delta_c, E_z,
                                                           self.cosmology.D_grid_z_full, self.cosmology.D_grid_full,
                                                           min_factor = 0.1, max_factor=20, return_profile_params=True)
                        Mvir_vec, R_vir_vec, r_s_vec, delta_char_vec = np.exp(np.interp(np.log(M_vec), np.log(M_vec_coarse), np.log(Mvir_coarse))),\
                                                                       np.exp(np.interp(np.log(M_vec), np.log(M_vec_coarse), np.log(R_vir_vec_coarse))),\
                                                                       np.exp(np.interp(np.log(M_vec), np.log(M_vec_coarse), np.log(r_s_vec_coarse))),\
                                                                       np.exp(np.interp(np.log(M_vec), np.log(M_vec_coarse), np.log(delta_char_vec_coarse)))
                        R_vir_vec_coarse = (3. * Mvir_coarse / (4. * np.pi * rho_m * Delta_vir))**(1./3.) # Mpc/h
                    else:
                        Mvir_coarse = find_M_vir_from_M_200c(M_vec_coarse, R_200c_coarse, 
                                                           rho_m, rho_crit_z,
                                                           Delta_vir, c_min, redshift, Om0, sigma_r,
                                                           self.cosmology.normalisation_cached, delta_c, E_z,
                                                           self.cosmology.D_grid_z_full, self.cosmology.D_grid_full,
                                                           min_factor = 0.1, max_factor=20)
                        Mvir_vec = np.exp(np.interp(np.log(M_vec), np.log(M_vec_coarse), np.log(Mvir_coarse)))
                else:
                    if return_profile_params:
                        Mvir_vec, R_vir_vec, r_s_vec, delta_char_vec = find_M_vir_from_M_200c(M_vec, R_200c, 
                                                                       rho_m, rho_crit_z,
                                                                       Delta_vir, c_min, redshift, Om0, sigma_r,
                                                                       self.cosmology.normalisation_cached, delta_c, E_z,
                                                                       self.cosmology.D_grid_z_full, self.cosmology.D_grid_full,
                                                                       min_factor = 0.1, max_factor=20, return_profile_params=True)
                        r_s_vec_coarse = r_s_vec
                        delta_char_vec_coarse = delta_char_vec
                        R_vir_vec_coarse = R_vir_vec
                        M_vec_coarse = M_vec
                    else:
                        Mvir_vec = find_M_vir_from_M_200c(M_vec, R_200c, 
                                                           rho_m, rho_crit_z,
                                                           Delta_vir, c_min, redshift, Om0, sigma_r,
                                                           self.cosmology.normalisation_cached, delta_c, E_z,
                                                           self.cosmology.D_grid_z_full, self.cosmology.D_grid_full,
                                                           min_factor = 0.1, max_factor=20)
                #Mvir_vec_with_h_units = Mvir_vec*self.h # Msol/h
                #R_vir = (3. * Mvir_vec / (4. * np.pi * rho_m * Delta_vir))**(1./3.) # Mpc/h
                #R_vir_with_h_units = (3. * Mvir_vec_with_h_units / (4. * np.pi * rho_m_with_h_units * Delta_vir))**(1./3.) # Mpc/h
                (sigma, dsigmadR_vir) = sigma_r.get_sigma_M(Mvir_vec, rho_m, get_deriv=True)
                R_lagrangian = sigma_r.R_eval   # the R that get_sigma_M actually used
                dM_dR_lagrangian = 4. * np.pi * rho_m * R_lagrangian**2
                dlnsigma2_dlnMvir = (Mvir_vec / sigma**2) * (2. * sigma * dsigmadR_vir) / dM_dR_lagrangian

                nu = delta_c / sigma
                p_st = 0.3
                q_st = 0.707
                A_st = np.sqrt(2.*q_st)/(np.sqrt(np.pi) + Gamma(0.5-p_st)/2**p_st)  # A ~ 0.2161
                func_sheth_tormen = A_st * nu * (1. + (q_st ** 0.5 * nu) ** (-2. * p_st)) * np.exp((-q_st * nu ** 2.) / 2.)

                hmf_vir = 0.5 * (rho_m / Mvir_vec**2) * func_sheth_tormen * np.abs(dlnsigma2_dlnMvir) # 1/M_vir dn/dlnM_vir = dn/dM_vir
                hmf = hmf_vir * np.gradient(Mvir_vec, M_vec) # dn/dM200c
                    
                M_eval = M_vec

                hmf    = hmf * 1e14
                M_eval = M_eval / 1e14

                if log == True:
                    hmf    = hmf * M_eval 
                    M_eval = np.log(M_eval)

            if self.hmf_type == "Tinker08":

                rho_m = self.rho_c_0*self.cosmology.cosmo_params["Om0"]

                if load_sigma_r is False:

                    k,ps = self.cosmology.power_spectrum.get_linear_power_spectrum(redshift)
                    k = jnp.asarray(k)
                    ps = jnp.asarray(ps)
                    sigma_r = sigma_R((k,ps),cosmology=self.cosmology)
                    sigma_r.get_derivative(type_deriv=self.type_deriv)

                elif load_sigma_r is True:

                    z_indices_key = np.array([float(index) for index in list(self.sigma_r_dict.keys())])
                    z_index = str(z_indices_key[np.argmin(np.abs(z_indices_key-redshift))])
                    sigma_r = self.sigma_r_dict[z_index]

                if save_sigma_r is True:

                    self.sigma_r_dict[str(redshift)] = sigma_r

                t0 = time.time()

                (sigma,dsigmadR) = sigma_r.get_sigma_M(M_vec,rho_m,get_deriv=True)

                self.sigma = sigma
                self.dsigmadR = dsigmadR
                self.R = sigma_r.R_eval

                dMdR = 4.*jnp.pi*rho_m*self.R**2

                if self.mass_definition[-1] == "c":

                    if self.cosmology.cnc_params["cosmology_tool"] == "cobaya_cosmo":

                        rescale = self.cosmology.Om(redshift)/(self.cosmology.H(redshift)/100.)**2

                    else:

                        rescale = self.cosmology.cosmo_params["Om0"]*(1.+redshift)**3/(self.cosmology.background_cosmology.H(redshift).value/(self.cosmology.cosmo_params["h"]*100.))**2

                elif self.mass_definition[-1] == "m":

                    rescale = 1

                Delta = float(self.mass_definition[0:-1])/rescale

                fsigma = f_sigma(sigma,redshift=redshift,hmf_type=self.hmf_type,
                Delta=Delta,mass_definition=self.mass_definition,
                other_params=self.other_params)
                self.fsigma = fsigma

                hmf = -fsigma*rho_m/M_vec/dMdR*dsigmadR/sigma
                M_eval = M_vec

                hmf = hmf*1e14
                M_eval = M_eval/1e14

                if log == True:

                    hmf = hmf*M_eval
                    M_eval = jnp.log(M_eval)

        elif self.hmf_calc == "hmf":

            self.massfunc_hmf.update(z=redshift)
            hmf = jnp.asarray(self.massfunc_hmf.dndm*1e14*self.h**4)
            M_eval = jnp.asarray(self.massfunc_hmf.m/self.h/1e14)

            hmf = jnp.interp(M_vec/1e14,M_eval,hmf)
            M_eval = M_vec/1e14

            if log == True:

                hmf = hmf*M_eval
                M_eval = jnp.log(M_eval)

        elif self.hmf_calc == "MiraTitan": #only works if log == True, note that returns a matrix instead of a vector

            t0 = time.time()

            if log == True:

                MT_emulator = self.extra_params["emulator"]

                M_vec = np.linspace(M_min,M_max,n_points)

                cosmology_emulator = {
                "h": self.h,
                "Ommh2": self.cosmology.cosmo_params["Om0"]*self.h**2,
                "Ombh2": self.cosmology.cosmo_params["Ob0"]*self.h**2,
                "Omnuh2": self.cosmology.Omega_nu*self.h**2,
                "sigma_8": self.cosmology.cosmo_params["sigma_8"],
                "n_s": self.cosmology.cosmo_params["n_s"],
                "w_0": -1.,
                "w_a": 0.
                }

                hmf = jnp.asarray(np.array(MT_emulator.predict(cosmology_emulator,redshift,M_vec*self.h))[0,:,:]*self.h**3)
                M_eval = jnp.log(M_vec/1e14)

                if volume_element == True:

                    for i in range(0,hmf.shape[0]):

                        hmf = hmf.at[i,:].set(hmf[i,:]*self.cosmology.background_cosmology.differential_comoving_volume(redshift[i]).value)

        if volume_element == True and self.hmf_calc != "MiraTitan":

            hmf = hmf*self.cosmology.background_cosmology.differential_comoving_volume(redshift).value

        if self.M_min_cutoff is not None:

            cutoff_mask = jnp.where(M_vec < self.M_min_cutoff, 0., 1.)
            if hmf.ndim == 2:
                hmf = hmf * cutoff_mask[jnp.newaxis, :]
            else:
                hmf = hmf * cutoff_mask

        if return_profile_params:     
            return M_eval,hmf,{
                'M_vec'      : M_vec_coarse,
                'rho_crit'   : rho_crit_z,
                'R_vir'      : R_vir_vec_coarse,
                'r_s'        : r_s_vec_coarse,
                'delta_char' : delta_char_vec_coarse,
                'rho_m'      : rho_m  # scalar, same for all M at this z
            }
        return M_eval,hmf


class sigma_R:
    """Computes the variance of the linear density field smoothed with a top-hat filter.

    Uses mcfit.TophatVar (FFTLog algorithm) with JAX backend for differentiable computation.
    Pre-built TophatVar objects can be passed via _tv0/_tv1 to avoid redundant constructor calls
    when the same k grid is used across multiple redshifts.
    """

    def __init__(self, ps, cosmology=None, deriv=0, _tv0=None, _tv1=None):

        self.cosmology = cosmology
        (self.k, self.pk) = ps

        # Use mcfit with JAX backend (no numpy/JAX conversions needed)
        if _tv0 is None:
            from mcfit import TophatVar
            _tv0 = TophatVar(np.asarray(self.k), lowring=True, deriv=0, backend='jax')

        self.R_vec, self.var_vec = _tv0(self.pk, extrap=True)
        self.sigma_vec = jnp.sqrt(self.var_vec)
        self._tv1 = _tv1

    def get_derivative(self, type_deriv="analytical"):

        if type_deriv == "analytical":

            if self._tv1 is None:
                from mcfit import TophatVar
                self._tv1 = TophatVar(np.asarray(self.k), lowring=True, deriv=1, backend='jax')

            _, dvar = self._tv1(self.pk * self.k, extrap=True)
            self.dsigma_vec = dvar / (2.0 * self.sigma_vec)

        elif type_deriv == "numerical":

            self.dsigma_vec = jnp.gradient(self.sigma_vec, self.R_vec)

    def get_sigma_M(self, M_vec, rho_m, get_deriv=False):

        R = (3. * M_vec / (4. * jnp.pi * rho_m))**(1./3.)
        self.R_eval = R

        sigma = jnp.interp(R, self.R_vec, self.sigma_vec)

        if get_deriv == False:

            ret = sigma

        elif get_deriv == True:

            dsigmadR = jnp.interp(R, self.R_vec, self.dsigma_vec)
            ret = (sigma, dsigmadR)

        return ret


def build_batch_sigma_fns(tv0, tv1, k_arr, type_deriv="analytical"):
    """Build cached vmapped functions for batch sigma computation.

    Call once per TophatVar pair (i.e., once per k grid). Returns functions
    that can be called repeatedly without re-tracing.

    Args:
        tv0: TophatVar(k, deriv=0, backend='jax') -- pre-built
        tv1: TophatVar(k, deriv=1, backend='jax') -- pre-built
        k_arr: (n_k,) JAX array of wavenumbers
        type_deriv: "analytical" or "numerical"

    Returns:
        (vmap_sigma_fn, vmap_interp_fn, R_vec) -- cached vmapped functions
    """
    R_vec = jnp.asarray(tv0.y)

    if type_deriv == "analytical":
        def _single_z(pk):
            _, var = tv0(pk, extrap=True)
            _, dvar = tv1(pk * k_arr, extrap=True)
            sigma_raw = jnp.sqrt(var)
            dsigma_raw = dvar / (2.0 * sigma_raw)
            return sigma_raw, dsigma_raw
    else:
        def _single_z(pk):
            _, var = tv0(pk, extrap=True)
            sigma_raw = jnp.sqrt(var)
            dsigma_raw = jnp.gradient(sigma_raw, R_vec)
            return sigma_raw, dsigma_raw

    vmap_sigma_fn = jax.jit(jax.vmap(_single_z))

    def _interp_to_M(sigma_row, dsigma_row, R_M):
        return jnp.interp(R_M, R_vec, sigma_row), jnp.interp(R_M, R_vec, dsigma_row)

    vmap_interp_fn = jax.jit(jax.vmap(_interp_to_M, in_axes=(0, 0, None)))

    return vmap_sigma_fn, vmap_interp_fn, R_vec


def batch_sigma_R_from_tophat(tv0, tv1, pk_batch, k_arr, M_vec, rho_m,
                               type_deriv="analytical",
                               _cached_fns=None):
    """Batch compute sigma(M) and dsigma/dR(M) for multiple power spectra via vmap.

    Args:
        tv0: TophatVar(k, deriv=0, backend='jax') -- pre-built
        tv1: TophatVar(k, deriv=1, backend='jax') -- pre-built
        pk_batch: (n_z, n_k) JAX array of power spectra
        k_arr: (n_k,) JAX array of wavenumbers
        M_vec: (n_M,) JAX array of masses
        rho_m: float, mean matter density
        type_deriv: "analytical" (FFTLog deriv=1) or "numerical" (jnp.gradient)
        _cached_fns: optional (vmap_sigma_fn, vmap_interp_fn, R_vec) from build_batch_sigma_fns

    Returns:
        sigma_matrix: (n_z, n_M)
        dsigma_matrix: (n_z, n_M)
        R_matrix: (n_z, n_M)
    """
    if _cached_fns is not None:
        vmap_sigma_fn, vmap_interp_fn, R_vec = _cached_fns
    else:
        vmap_sigma_fn, vmap_interp_fn, R_vec = build_batch_sigma_fns(
            tv0, tv1, k_arr, type_deriv)

    # Batch FFTLog transforms
    sigma_raw_batch, dsigma_raw_batch = vmap_sigma_fn(pk_batch)

    # Interpolate from R grid to mass-based R values
    R_M = (3. * M_vec / (4. * jnp.pi * rho_m))**(1./3.)
    sigma_matrix, dsigma_matrix = vmap_interp_fn(sigma_raw_batch, dsigma_raw_batch, R_M)
    R_matrix = jnp.broadcast_to(R_M[None, :], sigma_matrix.shape)

    return sigma_matrix, dsigma_matrix, R_matrix


#Delta is w.r.t. mean

def f_sigma(sigma, redshift=None, hmf_type="Tinker08", Delta=None, mass_definition="500c", other_params=None):

    params = hmf_params(hmf_type=hmf_type, mass_definition=mass_definition, other_params=other_params)

    if hmf_type == "Tinker08":

        alpha = 10.**(-(0.75/jnp.log10(Delta/75.))**1.2)

        A = params.get_param("A", Delta)*(1.+redshift)**(-0.14)
        a = params.get_param("a", Delta)*(1.+redshift)**(-0.06)
        b = params.get_param("b", Delta)*(1.+redshift)**(-alpha)
        c = params.get_param("c", Delta)

        f = A*((sigma/b)**(-a)+1.)*jnp.exp(-c/sigma**2)

    return f


def trapz(y, x):
    '''
    Pure python version of trapezoid rule.
    Taken from https://berkeley-stat159-f17.github.io/stat159-f17/lectures/09-intro-numpy/trapezoid..html
    '''
    s = 0
    for i in range(1, len(x)):
        s += (x[i]-x[i-1])*(y[i]+y[i-1])
    return s/2

def func_axionHMcode_D_z_unnorm(redshift, Om0, E_z):    
    #z_array = np.linspace(redshift, 100, 2000)
    #integrand = (1+z_array) / E_z(z_array, Om0, Ow0)**3

    #factor = 5 * Om0 / 2
    #D = factor * E_z(redshift) * trapz(integrand, z_array) # now with Numba trapz
    #return D

    integrand = lambda zp: (1 + zp) / E_z(zp)**3
    result, _ = integrate.quad(integrand, redshift, 100.)
    return 5 * Om0 / 2 * E_z(redshift) * result

def func_axionHMcode_D_z_unnorm_int(redshift, Om0, E_z):
    def integrand(y, x):
        E_x = E_z(x)
        E_y = E_z(y)
        return E_x / (1 + x) * (1 + y) / E_y**3
    G = 5 * Om0 / 2 * integrate.dblquad(
            integrand, redshift, 10000,
            lambda x: x, 10000)[0]
    return G

def func_axionHMcode_Delta_vir(redshift, Om0, G_a, E_z, g_a, version='dome'):   
    p_10 = -0.79
    p_11 = -10.17
    p_12 = 2.51
    p_13 = 6.51
    p_20 = -1.89
    p_21 = 0.38
    p_22 = 18.8
    p_23 = -15.87
    f_1 = p_10 + p_11*(1-g_a) + p_12*(1-g_a)**2 + p_13*(1-G_a*(1+redshift))
    f_2 = p_20 + p_21*(1-g_a) + p_22*(1-g_a)**2 + p_23*(1-G_a*(1+redshift))

    Omega_m_z = Om0 * (1+redshift)**3 / E_z(redshift)**2

    alpha_1 = 1
    alpha_2 = 2
    f_frac = 0.
    if version == 'dome':
        return 177.7 *(1+0.763*f_frac) * ( 1 + f_1*np.log10(Omega_m_z)**alpha_1 + f_2*np.log10(Omega_m_z)**alpha_2)
    else:
        return 177.7 * ( 1 + f_1*np.log10(Omega_m_z)**alpha_1 + f_2*np.log10(Omega_m_z)**alpha_2)


    return f

def func_axionHMcode_z_formation(redshift, Mvir_with_h_units, rho_m_with_h_units,
                                 Om0, sigma_r, normalisation, delta_c, E_z, f=0.01):
    def solve_single(M_single):
        sigma = sigma_r.get_sigma_M(f * M_single, rho_m_with_h_units, get_deriv=False)
        target = func_axionHMcode_D_z_unnorm(redshift, Om0, E_z) / normalisation * delta_c / sigma

        def func_find_root(x):
            return func_axionHMcode_D_z_unnorm(x, Om0, E_z) / normalisation - target

        f_lo = func_find_root(redshift)
        f_hi = func_find_root(100.)

        if f_lo * f_hi > 0.:
            return redshift   # no root found, z_f = z by definition
        return brentq(func_find_root, redshift, 100., xtol=1e-4)

    if isinstance(Mvir_with_h_units, (int, float)):
        return solve_single(Mvir_with_h_units)
    else:
        return np.array([solve_single(M) for M in Mvir_with_h_units])

def func_axionHMcode_z_formation_fast(redshift, M_vir_grid, rho_m, Om0,
                                      sigma_r, normalisation, delta_c, E_z,
                                      D_grid_z_full, D_grid_full, f=0.01):
    #Precomputes z_formation on a mass grid using vectorized sigma and a single precomputed D(z) interpolation table.
    # Precompute D(z)/D(0) on a z grid once
    mask = D_grid_z_full > redshift
    z_grid = D_grid_z_full[mask]
    D_grid = D_grid_full[mask]
    D_z = np.interp(redshift, D_grid_z_full, D_grid_full)

    # Vectorized sigma for all masses at once
    sigma_grid = sigma_r.get_sigma_M(f * M_vir_grid, rho_m, get_deriv=False)
    target_grid = D_z * delta_c / sigma_grid

    # For each mass, find z_f by interpolating the inverse D(z) table
    #def z_f_from_target(target):
    #    if target >= D_grid[0] or target <= D_grid[-1]:
    #        return redshift  # no root, z_f = z
    #    return np.interp(target, D_grid[::-1], z_grid[::-1])
    #z_f_grid = np.array([z_f_from_target(t) for t in target_grid])
    z_f_grid = np.where(
        (target_grid >= D_grid[0]) | (target_grid <= D_grid[-1]),
        redshift,
        np.interp(target_grid, D_grid[::-1], z_grid[::-1])
    )

    # Return a fast interpolator over log(Mvir)
    log_M_grid = np.log(M_vir_grid)
    def z_formation_interp(Mvir):
        return np.interp(np.log(Mvir), log_M_grid, z_f_grid)

    return z_formation_interp

def func_axionHMcode_delta_c(redshift, Om0, G_a, E_z, g_a, version='dome'):    
    p_10 = -0.0069
    p_11 = -0.0208
    p_12 = 0.0312
    p_13 = 0.0021
    p_20 = 0.0001
    p_21 = -0.0647
    p_22 = -0.0417
    p_23 = 0.0646
    f_1 = p_10 + p_11*(1-g_a) + p_12*(1-g_a)**2 + p_13*(1-G_a*(1+redshift))
    f_2 = p_20 + p_21*(1-g_a) + p_22*(1-g_a)**2 + p_23*(1-G_a*(1+redshift))

    Omega_m_z = Om0 * (1+redshift)**3 / E_z(redshift)**2

    alpha_1 = 1
    alpha_2 = 0
    f_frac = 0.
    if version == 'dome':
        return 1.686 *(1-0.041*f_frac)* ( 1 + f_1*np.log10(Omega_m_z)**alpha_1 + f_2*np.log10(Omega_m_z)**alpha_2)
    else:
        return 1.686 * ( 1 + f_1*np.log10(Omega_m_z)**alpha_1 + f_2*np.log10(Omega_m_z)**alpha_2)
'''
def find_M_vir_from_M_200c(M_vec_with_h, R_200c_with_h, rho_m_with_h, rho_crit_z_with_h,
                            Delta_vir, c_min, redshift, Om0, sigma_r, 
                            normalisation, delta_c, E_z, D_grid_z_full, D_grid_full,
                            min_factor = 0.5, max_factor=10, return_profile_params=False):
    def g(x):
        # NFW enclosed mass shape function
        return np.log(1. + x) - x / (1. + x)
    M_vir_grid = np.exp(np.linspace(
        np.log(min_factor * M_vec_with_h.min()),
        np.log(max_factor * M_vec_with_h.max()),
        300))
    z_formation_interp = func_axionHMcode_z_formation_fast(
        redshift, M_vir_grid, rho_m_with_h, Om0,
        sigma_r, normalisation, delta_c, E_z, 
        D_grid_z_full, D_grid_full, f=0.01)

    def residual_single(log_Mvir, M200c, R200c):
        Mvir = np.exp(log_Mvir)

        # concentration from axionHMcode c-M relation
        #z_f = func_axionHMcode_z_formation(redshift, Mvir, rho_m_with_h, Om0, G_a, sigma_r, normalisation, delta_c, f=0.01)
        z_f = z_formation_interp(Mvir)
        concentration = c_min * (1. + z_f) / (1. + redshift)

        # virial radius from M_vir definition (w.r.t. mean density at z=0)
        R_vir = (3. * Mvir / (4. * np.pi * rho_m_with_h * Delta_vir))**(1./3.)
        r_s   = R_vir / concentration

        # NFW characteristic density (delta_char * rho_mean)
        # rho(r) = delta_char * rho_m / ((r/r_s)(1+r/r_s)^2)
        # such that M(<R_vir) = M_vir by construction
        delta_char = Delta_vir * concentration**3 / (3. * g(concentration))

        # enclosed mass at R_200c
        x_200c = R200c / r_s
        M_enc  = 4. * np.pi * delta_char * rho_m_with_h * r_s**3 * g(x_200c)

        return M_enc - M200c

    def solve_single(M200c, R200c):
        # Bracket in log(M_vir): M_vir is usually within factor ~2 of M_200c
        log_lo = np.log(min_factor * M200c)
        log_hi = np.log(max_factor * M200c)

        # Check bracket is valid
        f_lo = residual_single(log_lo, M200c, R200c)
        f_hi = residual_single(log_hi, M200c, R200c)

        if f_lo * f_hi > 0.:
            print(f"Bracket failed for M200c {M200c}. Falling back to constant-ratio approximation")
            ratio = (200. * rho_crit_z_with_h / (Delta_vir * rho_m_with_h))
            return M200c * ratio

        log_Mvir = brentq(residual_single, log_lo, log_hi,
                          args=(M200c, R200c), xtol=1e-8, rtol=1e-6)
        #return np.exp(log_Mvir)
        Mvir = np.exp(log_Mvir)

        if return_profile_params:
            # recompute profile quantities at solution
            z_f        = z_formation_interp(Mvir)
            conc       = c_min * (1. + z_f) / (1. + redshift)
            R_vir_sol  = (3. * Mvir / (4. * np.pi * rho_m_with_h * Delta_vir))**(1./3.)
            r_s_sol    = R_vir_sol / conc
            delta_char_sol = Delta_vir * conc**3 / (3. * g(conc))

            return Mvir, R_vir_sol, r_s_sol, delta_char_sol

        return Mvir

    #solve_vec = np.vectorize(solve_single)
    #return solve_vec(M_vec_with_h, R_200c_with_h)
    #return np.vectorize(solve_single)(M_vec_with_h, R_200c_with_h)
    if return_profile_params:
        Mvir_vec, R_vir_vec, r_s_vec, delta_char_vec = np.vectorize(
            solve_single, otypes=[float, float, float, float]
        )(M_vec_with_h, R_200c_with_h)
        return Mvir_vec, R_vir_vec, r_s_vec, delta_char_vec
    else:
        return np.vectorize(solve_single)(M_vec_with_h, R_200c_with_h)
'''
@functools.partial(jax.jit, static_argnums=())
def _find_M_vir_jit(log_lo, log_hi, R_200c, rho_m, Delta_vir, c_min,
                     redshift, log_M_vir_grid, z_f_grid, n_iter=60):
    """JIT-compiled bisection for M_vir given M_200c array."""
    def g(x):
        return jnp.log(1. + x) - x / (1. + x)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        Mvir = jnp.exp(mid)
        z_f = jnp.interp(mid, log_M_vir_grid, z_f_grid)
        conc = c_min * (1. + z_f) / (1. + redshift)
        R_vir = (3. * Mvir / (4. * jnp.pi * rho_m * Delta_vir))**(1./3.)
        r_s = R_vir / conc
        delta_char = Delta_vir * conc**3 / (3. * g(conc))
        x_200c = R_200c / r_s
        M_enc = 4. * jnp.pi * delta_char * rho_m * r_s**3 * g(x_200c)
        lo = jnp.where(M_enc < jnp.exp(log_lo + log_hi - mid), mid, lo)  # wrong
        return (lo, hi), None

    def body_fixed(carry, _):
        lo, hi, M200c_arr = carry
        mid = 0.5 * (lo + hi)
        Mvir = jnp.exp(mid)
        z_f = jnp.interp(mid, log_M_vir_grid, z_f_grid)  # scalar per mass point — need vmap
        ...
        
@jax.jit
def _bisect_single(log_lo, log_hi, M200c, R200c, rho_m, Delta_vir, c_min,
                   redshift, log_M_vir_grid, z_f_grid):
    def g(x):
        return jnp.log(1. + x) - x / (1. + x)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        Mvir = jnp.exp(mid)
        z_f = jnp.interp(mid, log_M_vir_grid, z_f_grid)
        conc = c_min * (1. + z_f) / (1. + redshift)
        R_vir = (3. * Mvir / (4. * jnp.pi * rho_m * Delta_vir))**(1./3.)
        r_s = R_vir / conc
        delta_char = Delta_vir * conc**3 / (3. * g(conc))
        x_200c = R200c / r_s
        M_enc = 4. * jnp.pi * delta_char * rho_m * r_s**3 * g(x_200c)
        lo = jnp.where(M_enc < M200c, mid, lo)
        hi = jnp.where(M_enc < M200c, hi, mid)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (log_lo, log_hi), None, length=30)
    mid = 0.5 * (lo + hi)
    Mvir = jnp.exp(mid)
    z_f = jnp.interp(mid, log_M_vir_grid, z_f_grid)
    conc = c_min * (1. + z_f) / (1. + redshift)
    R_vir = (3. * Mvir / (4. * jnp.pi * rho_m * Delta_vir))**(1./3.)
    r_s = R_vir / conc
    delta_char = Delta_vir * conc**3 / (3. * g(conc))
    return Mvir, r_s, delta_char

_bisect_vmap = jax.jit(jax.vmap(_bisect_single,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None)))

def find_M_vir_from_M_200c(M_vec, R_200c, rho_m, rho_crit_z,
                            Delta_vir, c_min, redshift, Om0, sigma_r,
                            normalisation, delta_c, E_z, D_grid_z_full, D_grid_full,
                            min_factor=0.5, max_factor=10, return_profile_params=False):
    # Precompute z_formation grid (numpy, not JIT-able due to sigma_r)
    M_vir_grid = np.exp(np.linspace(
        np.log(min_factor * M_vec.min()),
        np.log(max_factor * M_vec.max()), 300))
    z_formation_interp = func_axionHMcode_z_formation_fast(
        redshift, M_vir_grid, rho_m, Om0, sigma_r,
        normalisation, delta_c, E_z, D_grid_z_full, D_grid_full)

    log_M_vir_grid = jnp.asarray(np.log(M_vir_grid))
    z_f_grid = jnp.asarray(np.array([z_formation_interp(m) for m in M_vir_grid]))

    log_lo = jnp.log(jnp.asarray(min_factor * M_vec))
    log_hi = jnp.log(jnp.asarray(max_factor * M_vec))
    M200c_jnp = jnp.asarray(M_vec)
    R200c_jnp = jnp.asarray(R_200c)

    Mvir_vec, r_s_vec, delta_char_vec = _bisect_vmap(
        log_lo, log_hi, M200c_jnp, R200c_jnp,
        float(rho_m), float(Delta_vir), float(c_min), float(redshift),
        log_M_vir_grid, z_f_grid)

    if return_profile_params:
        R_vir_vec = (3. * Mvir_vec / (4. * jnp.pi * rho_m * Delta_vir))**(1./3.)
        return np.asarray(Mvir_vec), np.asarray(R_vir_vec), np.asarray(r_s_vec), np.asarray(delta_char_vec)
    return np.asarray(Mvir_vec)
    
class hmf_params:

    def __init__(self, hmf_type="Tinker08", mass_definition="500c", other_params=None):

        self.hmf_type = hmf_type
        self.mass_definition = mass_definition
        self.other_params = other_params

        if self.hmf_type == "Tinker08":

            if other_params["interp_tinker"] == "log":

                Delta = jnp.log10(jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.]))

            elif other_params["interp_tinker"] == "linear":

                Delta = jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.])

            A = jnp.array([0.186,0.2,0.212,0.218,0.248,0.255,0.260,0.260,0.260])
            a = jnp.array([1.47,1.52,1.56,1.61,1.87,2.13,2.30,2.53,2.66])
            b = jnp.array([2.57,2.25,2.05,1.87,1.59,1.51,1.46,1.44,1.41])
            c = jnp.array([1.19,1.27,1.34,1.45,1.58,1.80,1.97,2.24,2.44])

            self.params = {"A":A,"b":b,"a":a,"c":c,"Delta":Delta}

    def get_param(self, param, Delta):

        if self.hmf_type == "Tinker08":

            if self.other_params["interp_tinker"] == "log":

                ret = jnp.interp(jnp.log10(Delta), self.params["Delta"], self.params[param])

            elif self.other_params["interp_tinker"] == "linear":

                ret = jnp.interp(Delta, self.params["Delta"], self.params[param])

        return ret


class constants:

    def __init__(self):

        self.c_light = 2.997924581e8
        self.G = 6.674*1e-11
        self.solar = 1.98855*1e30
        self.mpc = 3.08567758149137*1e22
        self.gamma =  self.G/self.c_light**2*self.solar/self.mpc
