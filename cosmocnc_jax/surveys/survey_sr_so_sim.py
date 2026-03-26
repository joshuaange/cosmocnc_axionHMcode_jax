import numpy as np
import jax
import jax.numpy as jnp
import cosmocnc_jax
import logging
from cosmocnc_jax.utils import simpson


# =====================================================================
# Pure JAX scaling relation functions (JIT-compatible, no class state)
# =====================================================================

def sr_q_so_sim_layer0(x0, prefactor_logy0, prefactor_M_500_to_theta,
                        sigma_sz_poly, alpha_szifi):
    """Pure JAX: q_so_sim layer 0: lnM -> log(y0/sigma)."""
    log_y0 = x0 * alpha_szifi + prefactor_logy0
    log_theta_500 = jnp.log(prefactor_M_500_to_theta) + x0 / 3.
    log_sigma_sz = jnp.polyval(sigma_sz_poly, log_theta_500)
    x1 = log_y0 - log_sigma_sz
    return x1, log_theta_500


def sr_q_so_sim_layer0_deriv(log_theta_500, sigma_sz_polyder, alpha_szifi):
    """Pure JAX: derivative of q_so_sim layer 0."""
    return alpha_szifi - jnp.polyval(sigma_sz_polyder, log_theta_500) / 3.


def sr_q_so_sim_layer1(x0, _pref_logy0, _pref_theta, dof):
    """Pure JAX: q_so_sim layer 1: log(s/n) -> q = sqrt(exp(x)^2 + dof).
    First 2 args after x0 are prefactors (passed through, unused here)."""
    return jnp.sqrt(jnp.exp(x0)**2 + dof)


def sr_q_so_sim_layer1_deriv(x0, dof):
    """Pure JAX: derivative of q_so_sim layer 1."""
    exp2 = jnp.exp(2. * x0)
    return exp2 / jnp.sqrt(exp2 + dof)


def sr_p_so_sim_layer0(x0, prefactor_lens, prefactor_M_500_to_theta_lensing,
                        sigma_lens_poly, a_lens, bias_cmblens):
    """Pure JAX: p_so_sim layer 0: lnM -> log(kappa/sigma_kappa)."""
    log_theta_500 = jnp.log(prefactor_M_500_to_theta_lensing) + x0 / 3.
    log_sigma = jnp.polyval(sigma_lens_poly, log_theta_500)
    x1 = jnp.log(prefactor_lens * a_lens * (0.1 * bias_cmblens)**(1./3.)) + x0 / 3. - log_sigma
    return x1


def sr_p_so_sim_layer1(x0, _pref_lens, _pref_theta):
    """Pure JAX: p_so_sim layer 1: log(kappa/sigma) -> kappa/sigma = exp(x).
    First 2 args after x0 are prefactors (passed through, unused here)."""
    return jnp.exp(x0)


def precompute_q_prefactors(E_z, H0, D_A, A_szifi, bias_sz, alpha_szifi):
    """Pure JAX: precompute q_so_sim prefactors at a single redshift."""
    h70 = H0 / 70.
    prefactor_logy0 = jnp.log(10.**(A_szifi) * E_z**2 *
                                (bias_sz / 3. * h70)**alpha_szifi / jnp.sqrt(h70))
    prefactor_M_500_to_theta = 6.997 * (H0 / 70.)**(-2./3.) * \
        (bias_sz / 3.)**(1./3.) * E_z**(-2./3.) * (500. / D_A)
    return prefactor_logy0, prefactor_M_500_to_theta


def precompute_p_prefactors(E_z, H0, D_A, D_CMB, D_l_CMB, rho_c, gamma,
                             bias_cmblens):
    """Pure JAX: precompute p_so_sim prefactors at a single redshift."""
    c = 3.
    r_s = (3. / 4. / rho_c / 500. / jnp.pi / c**3 * 1e15)**(1./3.)
    rho_0 = rho_c * 500. / 3. * c**3 / (jnp.log(1. + c) - c / (1. + c))
    Sigma_c = 1. / (4. * jnp.pi * D_A * D_l_CMB * gamma) * D_CMB
    R = 5. * c
    factor = r_s * rho_0 / Sigma_c
    convergence = 2. * (2. - 3. * R + R**3) / (3. * (-1. + R**2)**(3./2.))
    prefactor_lens = factor * convergence
    prefactor_M_500_to_theta_lensing = 6.997 * (H0 / 70.)**(-2./3.) * \
        (bias_cmblens / 3.)**(1./3.) * E_z**(-2./3.) * (500. / D_A)
    return prefactor_lens, prefactor_M_500_to_theta_lensing


def get_mean_p_so_sim(x0, prefactor_lens, prefactor_M_500_to_theta_lensing,
                       sigma_lens_poly, a_lens, bias_cmblens, sigma_intrinsic,
                       compute_var=False):
    """Pure JAX: compute mean (and optionally variance) of p_so_sim stacked observable."""
    log_theta_500 = jnp.log(prefactor_M_500_to_theta_lensing) + x0 / 3.
    log_sigma = jnp.polyval(sigma_lens_poly, log_theta_500)
    lnp_mean = jnp.log(prefactor_lens * a_lens * (0.1 * bias_cmblens)**(1./3.)) + x0 / 3. - log_sigma
    mean = jnp.exp(lnp_mean + sigma_intrinsic**2 * 0.5)
    var_total = jnp.where(
        compute_var,
        (jnp.exp(sigma_intrinsic**2) - 1.) * jnp.exp(2. * lnp_mean + sigma_intrinsic**2) + 1.,
        0.
    )
    return mean, var_total


def build_scatter_cov_layer0(sigma_lnq, sigma_lnp, corr_lnq_lnp, obs_config="qp"):
    """Pure JAX: build 2x2 layer-0 covariance matrix for q+p observables."""
    if obs_config == "qp":
        return jnp.array([[sigma_lnq**2, corr_lnq_lnp * sigma_lnq * sigma_lnp],
                           [corr_lnq_lnp * sigma_lnq * sigma_lnp, sigma_lnp**2]])
    elif obs_config == "q":
        return jnp.array([[sigma_lnq**2]])
    elif obs_config == "p":
        return jnp.array([[sigma_lnp**2]])
    return jnp.array([[0.]])


def build_scatter_cov_layer1(obs_config="qp"):
    """Pure JAX: build layer-1 covariance matrix (noise covariance = identity)."""
    if obs_config == "qp":
        return jnp.array([[1., 0.],
                           [0., 1.]])
    elif obs_config == "q":
        return jnp.array([[1.]])
    elif obs_config == "p":
        return jnp.array([[1.]])
    return jnp.array([[0.]])


def _tile_to_patches(params_tuple, n_patches):
    """Add leading n_patches dim to each param via broadcast_to (zero-copy).

    Scalar () -> (n_patches,). Array (d,) -> (n_patches, d). Etc.
    After p[patch_idx] inside JIT, the original shape is recovered.
    """
    return tuple(
        jnp.broadcast_to(jnp.asarray(p)[None], (n_patches,) + jnp.asarray(p).shape)
        for p in params_tuple
    )


class scaling_relations:

    def __init__(self,observable="q_mmf3",cnc_params=None,catalogue=None):

        self.logger = logging.getLogger(__name__)
        self.observable = observable
        self.cnc_params = cnc_params
        self.preprecompute = False
        self.catalogue = catalogue
        self.root_path = cosmocnc_jax.root_path


    def get_n_layers(self):

        observable = self.observable

        if observable == "p_so_sim_stacked":

            n_layers = 1

        else:

            n_layers = 2

        return n_layers

    def get_n_layers_stacked(self):

        return self.get_n_layers()

    def initialise_scaling_relation(self,cosmology=None):

        observable = self.observable
        self.const = cosmocnc_jax.constants()

        if observable == "p_so_sim_original" or observable == "p_so_sim_stacked":

            [theta_500_vec,sigma_lens_vec] = np.load(self.root_path + "data/so_sim_lensing_mf_noise.npy")
            theta_500_vec = theta_500_vec*180.*60./np.pi #in arcmin

            self.sigma_theta_lens_vec = np.zeros((1,2,len(theta_500_vec))) #first index is patch index, just 0
            self.sigma_theta_lens_vec[0,0,:] = theta_500_vec
            self.sigma_theta_lens_vec[0,1,:] = sigma_lens_vec

        if observable == "p_so_sim":

            [theta_500_vec,sigma_lens_vec] = np.load(self.root_path + "data/so_sim_lensing_mf_noise.npy")
            theta_500_vec = theta_500_vec*180.*60./np.pi #in arcmin

            x = np.log(theta_500_vec)
            y = np.log(sigma_lens_vec)
            self.sigma_lens_poly = jnp.asarray(np.polyfit(x,y,deg=3))

            sigma_sz_vec_eval = np.exp(np.polyval(self.sigma_lens_poly,x))

        if observable == "q_so_sim":

            theta_500_vec,sigma_sz_vec = np.load(self.root_path + "data/so_sim_sz_mf_noise.npy")

            self.theta_500_vec = theta_500_vec*180.*60./np.pi

            x = np.log(self.theta_500_vec)
            y = np.log(sigma_sz_vec)
            sigma_sz_poly_np = np.polyfit(x,y,deg=3)
            self.sigma_sz_poly = jnp.asarray(sigma_sz_poly_np)
            self.sigma_sz_polyder = jnp.asarray(np.polyder(sigma_sz_poly_np))

            self.skyfracs = [0.4] #from SO goals and forecasts paper

            #False detection pdf

            q_vec = np.linspace(5.,10.,self.cnc_params["n_points"])
            pdf_fd = np.exp(-(q_vec-3.)**2/1.5**2)
            pdf_fd = pdf_fd/simpson(pdf_fd,x=q_vec)
            self.pdf_false_detection = [q_vec,pdf_fd]

    def precompute_scaling_relation(self,params=None,other_params=None,patch_index=0):

        observable = self.observable
        self.params = params

        if observable == "p_so_sim" or observable == "p_so_sim_stacked":

            H0 = other_params["H0"]
            E_z = other_params["E_z"]
            D_A = other_params["D_A"]
            D_CMB = other_params["D_CMB"]
            D_l_CMB = other_params["D_l_CMB"]
            rho_c = other_params["rho_c"] # cosmology.critical_density(z_obs).value*1000.*mpc**3/solar
            gamma = self.const.gamma

            c = 3.
            r_s = (3./4./rho_c/500./jnp.pi/c**3*1e15)**(1./3.)
            rho_0 = rho_c*500./3.*c**3/(jnp.log(1.+c)-c/(1.+c))
            Sigma_c = 1./(4.*jnp.pi*D_A*D_l_CMB*gamma)*D_CMB
            R = 5.*c
            factor = r_s*rho_0/Sigma_c
            convergence = 2.*(2.-3.*R+R**3)/(3.*(-1.+R**2)**(3./2.))

            self.prefactor_lens = factor*convergence
            self.prefactor_M_500_to_theta_lensing = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_cmblens"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

        elif observable == "q_so_sim":

            E_z = other_params["E_z"]
            h70 = other_params["H0"]/70.
            H0 = other_params["H0"]
            D_A = other_params["D_A"]

            A_szifi = self.params["A_szifi"]
            self.prefactor_logy0 = jnp.log(10.**(A_szifi)*E_z**2*(self.params["bias_sz"]/3.*h70)**self.params["alpha_szifi"]/jnp.sqrt(h70))
            self.prefactor_M_500_to_theta = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_sz"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

    def eval_scaling_relation(self,x0,layer=0,patch_index=0,other_params=None):

        observable = self.observable

        if observable == "p_so_sim":

            if layer == 0:

                log_theta_500_lensing = jnp.log(self.prefactor_M_500_to_theta_lensing) + x0/3.
                log_sigma = jnp.polyval(self.sigma_lens_poly,log_theta_500_lensing)

                x1 = jnp.log(self.prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma

            elif layer == 1:

                x1 = jnp.exp(x0)

        if observable == "p_so_sim_stacked":

            if layer == 0:

                log_theta_500_lensing = jnp.log(self.prefactor_M_500_to_theta_lensing) + x0/3.
                log_sigma = jnp.polyval(self.sigma_lens_poly,log_theta_500_lensing)

                x1 = jnp.log(self.prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma

        if observable == "q_so_sim":

            if layer == 0:

                log_y0 = x0*self.params["alpha_szifi"] + self.prefactor_logy0
                log_theta_500 = jnp.log(self.prefactor_M_500_to_theta) + x0/3.
                self.log_theta_500 = log_theta_500
                log_sigma_sz = jnp.polyval(self.sigma_sz_poly,log_theta_500)
                x1 = log_y0 - log_sigma_sz

            if layer == 1:

                x1 = jnp.sqrt(jnp.exp(x0)**2+self.params["dof"])

        self.x1 = x1

        return x1

    def eval_derivative_scaling_relation(self,x0,layer=0,patch_index=0,scalrel_type_deriv="analytical"):

        observable = self.observable
        dx1_dx0 = None

        if scalrel_type_deriv == "analytical":

            if observable == "q_so_sim":

                if layer == 0:

                    dx1_dx0 = self.params["alpha_szifi"] - jnp.polyval(self.sigma_sz_polyder,self.log_theta_500)/3.

                if layer == 1:

                    dof = self.params["dof"]
                    exp = jnp.exp(2.*x0)
                    dx1_dx0 = exp/jnp.sqrt(exp+dof)

        if scalrel_type_deriv == "numerical" or dx1_dx0 is None: #must always be computed strictly after executing self.eval_scaling_relation()

            dx1_dx0 = jnp.gradient(self.x1,x0)

        return dx1_dx0

    def eval_scaling_relation_no_precompute(self,x0,layer=0,patch_index=0,other_params=None,params=None):

        self.params = params
        self.other_params = other_params
        observable = self.observable

        if observable == "q_so_sim":

            if layer == 0:

                E_z = other_params["E_z"]
                H0 = other_params["H0"]
                h70 = H0/70.
                D_A = other_params["D_A"]
                A_szifi = self.params["A_szifi"]

                prefactor_logy0 = jnp.log(10.**(A_szifi)*E_z**2*(self.params["bias_sz"]/3.*h70)**self.params["alpha_szifi"]/jnp.sqrt(h70))
                log_y0 = prefactor_logy0 + x0*self.params["alpha_szifi"]

                prefactor_M_500_to_theta = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_sz"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

                log_theta_500 = jnp.log(prefactor_M_500_to_theta) + x0/3.
                log_sigma_sz = jnp.polyval(self.sigma_sz_poly,log_theta_500)
                x1 = log_y0 - log_sigma_sz

            elif layer == 1:

                x1 = jnp.sqrt(jnp.exp(x0)**2+self.params["dof"])

        if observable == "p_so_sim" or observable == "p_so_sim_stacked":

            if layer == 0:

                H0 = other_params["H0"]
                E_z = other_params["E_z"]
                D_A = other_params["D_A"]
                D_CMB = other_params["D_CMB"]
                D_l_CMB = other_params["D_l_CMB"]
                rho_c = other_params["rho_c"] # cosmology.critical_density(z_obs).value*1000.*mpc**3/solar
                gamma = self.const.gamma

                c = 3.
                r_s = (3./4./rho_c/500./jnp.pi/c**3*1e15)**(1./3.)
                rho_0 = rho_c*500./3.*c**3/(jnp.log(1.+c)-c/(1.+c))
                Sigma_c = 1./(4.*jnp.pi*D_A*D_l_CMB*gamma)*D_CMB
                R = 5.*c
                factor = r_s*rho_0/Sigma_c
                convergence = 2.*(2.-3.*R+R**3)/(3.*(-1.+R**2)**(3./2.))

                prefactor_lens = factor*convergence
                prefactor_M_500_to_theta_lensing = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_cmblens"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)


                log_theta_500_lensing = jnp.log(prefactor_M_500_to_theta_lensing) + x0/3.
                log_sigma = jnp.polyval(self.sigma_lens_poly,log_theta_500_lensing)

                x1 = jnp.log(prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma

            elif layer == 1:

                x1 = jnp.exp(x0)

        return x1


    def get_mean(self,x0,patch_index=0,scatter=None,compute_var=False,other_params=None):

        if self.observable == "p_so_sim":

            log_theta_500_lensing = jnp.log(self.prefactor_M_500_to_theta_lensing) + x0/3.
            log_sigma = jnp.polyval(self.sigma_lens_poly,log_theta_500_lensing)

            lnp_mean = jnp.log(self.prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma
            sigma_intrinsic = jnp.sqrt(scatter.get_cov(observable1=self.observable,
                                                         observable2=self.observable,
                                                         layer=0,patch1=patch_index,patch2=patch_index))

            mean = jnp.exp(lnp_mean + sigma_intrinsic**2*0.5)
            ret = mean

            if compute_var == True:

                var_intrinsic = (jnp.exp(sigma_intrinsic**2)-1.)*jnp.exp(2.*lnp_mean+sigma_intrinsic**2)
                var_total = var_intrinsic + 1.
                ret = [mean,var_total]

        return ret

    def get_cutoff(self,layer=0):

        if self.observable == "q_so_sim":

            if layer == 0:

                cutoff = -jnp.inf

            elif layer == 1:

                cutoff = self.params["q_cutoff"]

        return cutoff

    # =================================================================
    # Factory methods: return pure JAX functions for JIT closure capture.
    # These allow cnc.py to build JIT-compiled kernels without knowing
    # which specific observable it is working with.
    # =================================================================

    def get_layer_fn(self, layer):
        """Return a pure JAX function for this observable's forward layer.

        Layer 0 functions return (x1, aux) where aux may be needed by derivative.
        Layer 1 functions return x1 only.
        The caller must pass: (x0, *prefactors, *layer_sr_params) as positional args.
        """
        obs = self.observable
        if obs == "q_so_sim":
            if layer == 0:
                return sr_q_so_sim_layer0
            elif layer == 1:
                return sr_q_so_sim_layer1
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            if layer == 0:
                return sr_p_so_sim_layer0
            elif layer == 1:
                return sr_p_so_sim_layer1
        raise ValueError(f"No layer function for observable={obs}, layer={layer}")

    def get_layer_returns_aux(self, layer):
        """Whether the layer function returns (x1, aux) instead of just x1."""
        if self.observable == "q_so_sim" and layer == 0:
            return True
        return False

    def get_layer_deriv_fn(self, layer):
        """Return a pure JAX analytical derivative function, or None."""
        obs = self.observable
        if obs == "q_so_sim":
            if layer == 0:
                return sr_q_so_sim_layer0_deriv
            elif layer == 1:
                return sr_q_so_sim_layer1_deriv
        return None

    def get_layer_deriv_uses_aux(self, layer):
        """Whether the derivative function takes aux (from layer fn) as first arg."""
        if self.observable == "q_so_sim" and layer == 0:
            return True
        return False

    def get_prefactor_fn(self):
        """Return the pure JAX prefactor function for this observable."""
        obs = self.observable
        if obs == "q_so_sim":
            return precompute_q_prefactors
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            return precompute_p_prefactors
        raise ValueError(f"No prefactor function for observable={obs}")

    def get_prefactor_vmap_axes(self):
        """Return in_axes for vmapping the prefactor function over redshift."""
        obs = self.observable
        if obs == "q_so_sim":
            return (0, None, 0, None, None, None)
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            return (0, None, 0, None, 0, 0, None, None)
        raise ValueError(f"No prefactor vmap axes for observable={obs}")

    def get_prefactor_args(self, cosmo_quantities, sr_params):
        """Build the argument tuple for the prefactor function.

        cosmo_quantities: dict with keys E_z, H0, D_A, D_CMB, D_l_CMB, rho_c, gamma.
                          Values may be scalar or array (e.g. over redshift).
        sr_params: dict of scaling relation parameters.
        """
        obs = self.observable
        if obs == "q_so_sim":
            return (cosmo_quantities["E_z"], cosmo_quantities["H0"],
                    cosmo_quantities["D_A"],
                    jnp.float64(sr_params["A_szifi"]),
                    jnp.float64(sr_params["bias_sz"]),
                    jnp.float64(sr_params["alpha_szifi"]))
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            return (cosmo_quantities["E_z"], cosmo_quantities["H0"],
                    cosmo_quantities["D_A"], cosmo_quantities["D_CMB"],
                    cosmo_quantities["D_l_CMB"], cosmo_quantities["rho_c"],
                    cosmo_quantities["gamma"],
                    jnp.float64(sr_params["bias_cmblens"]))
        raise ValueError(f"No prefactor args for observable={obs}")

    def get_n_prefactors(self):
        """Return number of prefactor arrays returned by the prefactor function."""
        if self.observable == "q_so_sim":
            return 2
        elif self.observable in ("p_so_sim", "p_so_sim_stacked"):
            return 2
        return 0

    def get_prefactor_fn_unified(self):
        """Return prefactor function with standardized signature.

        Returns a pure JAX function:
            fn(E_z, D_A, D_l_CMB, rho_c, H0, D_CMB, gamma, *sr_params) -> prefactors

        All cosmo quantities are passed (function uses what it needs).
        sr_params come from get_prefactor_sr_params().
        """
        obs = self.observable
        if obs == "q_so_sim":
            def fn(E_z, D_A, D_l_CMB, rho_c, H0, D_CMB, gamma, z_val,
                   A_szifi, bias_sz, alpha_szifi):
                return precompute_q_prefactors(E_z, H0, D_A,
                                                A_szifi, bias_sz, alpha_szifi)
            return fn
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            def fn(E_z, D_A, D_l_CMB, rho_c, H0, D_CMB, gamma, z_val,
                   bias_cmblens):
                return precompute_p_prefactors(E_z, H0, D_A, D_CMB,
                                                D_l_CMB, rho_c, gamma,
                                                bias_cmblens)
            return fn
        raise ValueError(f"No unified prefactor function for observable={obs}")

    def get_prefactor_sr_params(self, sr_params):
        """Return the SR params needed for prefactor computation as a tuple.

        These are passed as *sr_params to the unified prefactor function.
        """
        obs = self.observable
        if obs == "q_so_sim":
            return (jnp.float64(sr_params["A_szifi"]),
                    jnp.float64(sr_params["bias_sz"]),
                    jnp.float64(sr_params["alpha_szifi"]))
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            return (jnp.float64(sr_params["bias_cmblens"]),)
        return ()

    def get_n_prefactor_sr_params(self):
        """Return number of SR params passed to the unified prefactor function."""
        obs = self.observable
        if obs == "q_so_sim":
            return 3
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            return 1
        return 0

    def get_layer_sr_params(self, layer, sr_params):
        """Return the tuple of non-prefactor SR args for a layer function call.

        These are the arguments after x0 and the prefactors.
        """
        obs = self.observable
        if obs == "q_so_sim":
            if layer == 0:
                return (self.sigma_sz_poly, jnp.float64(sr_params["alpha_szifi"]))
            elif layer == 1:
                return (jnp.float64(sr_params["dof"]),)
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            if layer == 0:
                return (self.sigma_lens_poly,
                        jnp.float64(sr_params.get("a_lens", 1.)),
                        jnp.float64(sr_params.get("bias_cmblens", 1.)))
            elif layer == 1:
                return ()
        raise ValueError(f"No layer SR params for observable={obs}, layer={layer}")

    def get_layer_deriv_sr_params(self, layer, sr_params):
        """Return the tuple of SR args for the analytical derivative function.

        For q_so_sim layer 0: (sigma_sz_polyder, alpha_szifi)
        For q_so_sim layer 1: (dof,)
        """
        obs = self.observable
        if obs == "q_so_sim":
            if layer == 0:
                return (self.sigma_sz_polyder, jnp.float64(sr_params["alpha_szifi"]))
            elif layer == 1:
                return (jnp.float64(sr_params["dof"]),)
        return ()

    def get_scatter_sigma(self, sr_params):
        """Return the intrinsic scatter sigma for this observable at layer 0."""
        obs = self.observable
        if obs == "q_so_sim":
            return float(sr_params.get("sigma_lnq_szifi", 0.))
        elif obs in ("p_so_sim", "p_so_sim_stacked"):
            return float(sr_params.get("sigma_lnp", 0.))
        return 0.

    def get_mean_fn(self):
        """Return the pure JAX stacked mean function, or None."""
        if self.observable in ("p_so_sim", "p_so_sim_stacked"):
            return get_mean_p_so_sim
        return None

    def get_mean_fn_sr_params(self, sr_params):
        """Return the non-x0/non-prefactor args for the stacked mean function."""
        if self.observable in ("p_so_sim", "p_so_sim_stacked"):
            return (self.sigma_lens_poly,
                    jnp.float64(sr_params.get("a_lens", 1.)),
                    jnp.float64(sr_params.get("bias_cmblens", 1.)))
        return ()


class scatter:

    def __init__(self,params=None,catalogue=None):

        self.params = params
        self.catalogue = catalogue

    def get_cov(self,observable1=None,observable2=None,patch1=0,patch2=0,layer=0,other_params=None):

        if layer == 0:

            if observable1 == "p_so_sim" and observable2 == "p_so_sim":

                cov = self.params["sigma_lnp"]**2

            elif ((observable1 == "p_so_sim" and observable2 == "q_so_sim") or
                  (observable1 == "q_so_sim" and observable2 == "p_so_sim")):

                cov =  self.params["corr_lnq_lnp"]*self.params["sigma_lnq_szifi"]*self.params["sigma_lnp"]

            elif (observable1 == "q_so_sim" and observable2 == "q_so_sim"):

                cov = self.params["sigma_lnq_szifi"]**2

            else:

                cov = 0.

        elif layer == 1:

            if observable1 == "p_so_sim" and observable2 == "p_so_sim":

                cov = 1.

            elif (observable1 == "q_so_sim" and observable2 == "q_so_sim"):

                cov = 1

            else:

                cov = 0.

        return cov
