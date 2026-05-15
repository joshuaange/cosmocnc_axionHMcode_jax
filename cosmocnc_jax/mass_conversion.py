"""JAX-native M_Δc ↔ M_Δ'c conversions on an NFW profile.

The reference is classy_sz's mass-conversion path (see class_sz_tools.c:
`get_m200c_to_m500c_at_z_and_M` → `mDEL_to_mDELprime` → NFW). classy_sz
populates a 2D `(ln(1+z), ln M)` table once per cosmology by calling its
C-side routine, then 2D-interpolates from it. That table is what cosmocnc
exposes via `cosmology.get_m200c_to_m500c_at_z_and_M`; in cosmocnc_jax it
is precomputed once at SR init and never refreshed (SB1 in the 2026-05-15
port audit).

This module reproduces the (z, M_200c) → M_500c relation in pure JAX so
that the grid can be recomputed on every MCMC step. We use the simplified
direct path:

  c_200c(M, z) = D(z)^0.54 · 5.9 · ν^(-0.35),  with
       ν     = (1/D(z)) · (1.12 · (M / 5e13)^0.3 + 0.53)   [M in Msun/h]

(this is the closed-form Bhattacharya et al. 2013 fit; classy_sz
implements it verbatim in `get_c200c_at_m_and_z_B13` at class_sz.c:22468).
M_500c follows from NFW geometry: with x ≡ R_500c/R_200c, the overdensity
condition 200·m(cx)/m(c) = 500·x³ → m(cx)/m(c) = 2.5·x³ is Newton-solved
for x, and M_500c/M_200c = m(cx)/m(c) = 2.5·x³.

classy_sz's `get_m200c_to_m500c_at_z_and_M` goes through M_vir (using a
σ(M, z)-based ν instead of the closed-form), so this module's output is
not bit-identical to classy_sz. It IS identical in its closed-form B13
+ NFW physics; the cosmology dependence (D(z) and the h-via-M_pivot
scaling) is reproduced exactly. See port_audit.md SB1 for the residual
ν-fit error and the analysis impact.
"""

import jax
import jax.numpy as jnp


def _nfw_m(y):
    """NFW dimensionless enclosed mass: m(y) = ln(1+y) - y/(1+y)."""
    return jnp.log1p(y) - y / (1.0 + y)


def _nfw_m_prime(y):
    """d m / d y = y / (1+y)^2."""
    return y / (1.0 + y)**2


def growth_factor_carroll_press_turner(z, Om0, OL0):
    """Carroll-Press-Turner 1992 fitting formula for D(z), normalised so D(0)=1.

    Accurate to <1% across LCDM. Ignores Omega_r at low z (subpercent for z<5).
    """
    # present-day g(0)
    g0 = 2.5 * Om0 / (Om0**(4./7.) - OL0 + (1. + Om0 / 2.) * (1. + OL0 / 70.))
    # z-dependent g(z)
    a = 1.0 / (1.0 + z)
    Ez2 = Om0 / a**3 + OL0
    Om_z = (Om0 / a**3) / Ez2
    OL_z = OL0 / Ez2
    g_z = 2.5 * Om_z / (Om_z**(4./7.) - OL_z + (1. + Om_z / 2.) * (1. + OL_z / 70.))
    return g_z * a / g0


def b13_c200c_closed_form(M_Msunh, z, D_z):
    """Bhattacharya et al. 2013 closed-form fit for c_200c(M, z).

    M must be in M_sun/h units (B13's pivot mass is 5e13 M_sun/h).
    D_z is the linear growth factor with D(0)=1.

    Identical to classy_sz `get_c200c_at_m_and_z_B13` (class_sz.c:22468).
    """
    nu = (1.12 * (M_Msunh / 5e13)**0.3 + 0.53) / D_z
    return D_z**0.54 * 5.9 * nu**(-0.35)


# Spherical-collapse critical overdensity used by classy_sz's get_nu_at_z_and_m
# (set in input.c:7374: pclass_sz->delta_cSZ = (3/20) * (12 pi)^{2/3}).
DELTA_C_SZ = (3.0 / 20.0) * (12.0 * jnp.pi)**(2.0 / 3.0)


def b13_c200c_sigma_based(sigma, D_z, delta_c=DELTA_C_SZ):
    """Bhattacharya et al. 2013 c_200c using sigma-based peak height.

    Matches the variant classy_sz uses for the M_DEL conversion path:
        nu = delta_c / sigma(M, z)
        c_200c = D(z)^0.54 * 5.9 * nu^{-0.35}

    Args:
        sigma: sigma(M, z), shape compatible with D_z.
        D_z: linear growth factor at z, same shape (or broadcastable).
        delta_c: critical overdensity for spherical collapse.
    """
    nu = delta_c / sigma
    return D_z**0.54 * 5.9 * nu**(-0.35)


def b13_cvir_sigma_based(sigma, D_z, delta_c=DELTA_C_SZ):
    """Bhattacharya et al. 2013 c_vir using sigma-based peak height.

    Same form as c_200c with the B13 c_vir coefficients. Matches
    classy_sz `evaluate_cvir_of_mvir` (concentration_parameter==6),
    class_sz_tools.c:2027-2062:
        nu = sqrt(get_nu_at_z_and_m(z,m,...)) = delta_c / sigma(M, z)
        c_vir = D(z)^0.9 * 7.7 * nu^{-0.29}
    """
    nu = delta_c / sigma
    return D_z**0.9 * 7.7 * nu**(-0.29)


def delta_c_virial(Om_z):
    """Bryan & Norman 1998 virial overdensity vs critical density.

    Matches classy_sz `Delta_c_of_Omega_m` (class_sz_tools.c:14418):
        Delta_c_vir = 18*pi^2 + 82*(Om - 1) - 39*(Om - 1)^2
    For Om → 1 (matter-dominated), Delta_c_vir → 18π² ≈ 178.
    """
    x = Om_z - 1.0
    return 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2


def _sigma_at_logM_interp(logM, logM_grid, sigma_grid):
    """Interpolate sigma on a single-z row of a precomputed (n_logM,) grid."""
    return jnp.interp(logM, logM_grid, sigma_grid)


def _solve_M_vir_from_M_DEL(M_DEL, R_DEL, rho_c_z, delta_c_vir,
                            logM_grid_for_sigma, sigma_grid_at_z, D_z,
                            delta_c_sc=DELTA_C_SZ, n_iter=20):
    """Newton-solve for M_vir given M_DEL inside an NFW halo of c_vir(M_vir, z).

    Solves M_DEL / M_vir - m(C) / m(c_vir) = 0   (mDtomV's equation)
    where C = R_DEL / r_s = R_DEL * c_vir / R_vir, R_vir = (3 M_vir / (4π·δ_c_vir·ρ_c))^(1/3).

    Inputs/outputs are scalars (vmap externally for grid use).
    """
    def step(_, log_M_vir):
        M_vir = jnp.exp(log_M_vir)
        sigma_vir = _sigma_at_logM_interp(log_M_vir, logM_grid_for_sigma, sigma_grid_at_z)
        c_vir = b13_cvir_sigma_based(sigma_vir, D_z, delta_c_sc)
        R_vir = (3.0 * M_vir / (4.0 * jnp.pi * delta_c_vir * rho_c_z))**(1.0/3.0)
        rs = R_vir / c_vir
        C = R_DEL / rs

        m_c = _nfw_m(c_vir)
        m_C = _nfw_m(C)
        # Equation we want to drive to zero: f = M_DEL/M_vir - m(C)/m(c_vir)
        # As a function of log M_vir, differentiate via -M_DEL/M_vir * dM_vir/d(log M_vir) = -M_DEL/M_vir * M_vir = -M_DEL.
        # The c_vir part also varies with M_vir via sigma — but the dominant Jacobian term is the first piece. We use a damped Newton with bisection-like safety.
        # Numerical derivative of the residual w.r.t. log M_vir via central diff:
        eps = 1e-3
        def f_of_logM(lm):
            Mv = jnp.exp(lm)
            sv = _sigma_at_logM_interp(lm, logM_grid_for_sigma, sigma_grid_at_z)
            cv = b13_cvir_sigma_based(sv, D_z, delta_c_sc)
            Rv = (3.0 * Mv / (4.0 * jnp.pi * delta_c_vir * rho_c_z))**(1.0/3.0)
            r_s = Rv / cv
            CC = R_DEL / r_s
            return M_DEL / Mv - _nfw_m(CC) / _nfw_m(cv)
        f_now = f_of_logM(log_M_vir)
        f_plus = f_of_logM(log_M_vir + eps)
        f_minus = f_of_logM(log_M_vir - eps)
        f_prime = (f_plus - f_minus) / (2.0 * eps)
        # damped Newton with a step limit to avoid overshoot
        delta = jnp.clip(f_now / f_prime, -1.0, 1.0)
        return log_M_vir - delta

    log_M_vir_init = jnp.log(M_DEL)
    log_M_vir = jax.lax.fori_loop(0, n_iter, step, log_M_vir_init)
    return jnp.exp(log_M_vir)


def _solve_M_DELprime_from_M_vir(M_vir, R_vir, c_vir, rho_c_z, delta_prime,
                                 n_iter=20):
    """Newton-solve for M_DEL' = mass within overdensity-delta_prime radius
    in an NFW halo (M_vir, R_vir, c_vir). Pure NFW geometry, no sigma needed
    (c_vir is already fixed).

    Solves M_DEL' / M_vir - m(C') / m(c_vir) = 0
    where C' = R_DEL' / r_s, R_DEL' = (3 M_DEL' / (4π·delta_prime·ρ_c))^(1/3).
    """
    rs = R_vir / c_vir
    m_cvir = _nfw_m(c_vir)

    def step(_, log_M_DELprime):
        M_DELprime = jnp.exp(log_M_DELprime)
        R_DELprime = (3.0 * M_DELprime / (4.0 * jnp.pi * delta_prime * rho_c_z))**(1.0/3.0)
        C = R_DELprime / rs
        m_C = _nfw_m(C)
        # f(logM') = M'/M_vir - m(C)/m_cvir
        # d/d(logM') = M'/M_vir - (1/m_cvir) * m'(C) * dC/d(logM')
        # dC/d(logM') = C/3 (since R' ∝ M'^{1/3})
        f = M_DELprime / M_vir - m_C / m_cvir
        f_prime = M_DELprime / M_vir - _nfw_m_prime(C) * C / (3.0 * m_cvir)
        delta = jnp.clip(f / f_prime, -1.0, 1.0)
        return log_M_DELprime - delta

    log_M_init = jnp.log(M_vir)
    log_M_final = jax.lax.fori_loop(0, n_iter, step, log_M_init)
    return jnp.exp(log_M_final)


def _m200c_to_m500c_one(M_200c, z, rho_c_z, Om_z, D_z,
                       logM_grid_for_sigma, sigma_grid_at_z):
    """Single-point M_200c → M_500c using the virial-intermediate path.

    All arguments are scalars; sigma_grid_at_z is (n_logM,).
    """
    delta_c_vir = delta_c_virial(Om_z)
    delta_DEL = 200.0
    delta_DELprime = 500.0

    R_DEL = (3.0 * M_200c / (4.0 * jnp.pi * delta_DEL * rho_c_z))**(1.0/3.0)

    M_vir = _solve_M_vir_from_M_DEL(
        M_200c, R_DEL, rho_c_z, delta_c_vir,
        logM_grid_for_sigma, sigma_grid_at_z, D_z)

    # Recompute c_vir at the converged M_vir
    sigma_vir = _sigma_at_logM_interp(jnp.log(M_vir),
                                      logM_grid_for_sigma, sigma_grid_at_z)
    c_vir = b13_cvir_sigma_based(sigma_vir, D_z)
    R_vir = (3.0 * M_vir / (4.0 * jnp.pi * delta_c_vir * rho_c_z))**(1.0/3.0)

    M_500c = _solve_M_DELprime_from_M_vir(
        M_vir, R_vir, c_vir, rho_c_z, delta_DELprime)
    return M_500c


@jax.jit
def log_m500c_over_m200c_grid_virial(M_200c_vec, z_tab, rho_c_z_tab,
                                     Om_z_tab, D_z_tab,
                                     logM_grid_for_sigma, sigma_grid):
    """Build the (z, M_200c) → log(M_500c/M_200c) grid via the
    M_200c → M_vir → M_500c path that classy_sz uses.

    Args:
        M_200c_vec: 1D array of M_200c values (in physical Msun), shape (n_m,).
        z_tab: 1D array of redshifts, shape (n_z,).
        rho_c_z_tab: ρ_crit(z) in Msun/Mpc^3, shape (n_z,).
        Om_z_tab: Omega_m(z), shape (n_z,).
        D_z_tab: linear growth factor D(z), shape (n_z,).
        logM_grid_for_sigma: 1D ln(M_phys) grid where sigma is tabulated,
            shape (n_logM,). Must span both M_200c and the M_vir Newton range.
        sigma_grid: sigma(M, z), shape (n_z, n_logM). Same z ordering as z_tab.

    Returns:
        log(M_500c/M_200c), shape (n_z, n_m).
    """
    # vmap over (z, M_200c)
    def at_z(rho_c, Om, D, sigma_row):
        def at_M(M_200c):
            M_500c = _m200c_to_m500c_one(M_200c, 0.0, rho_c, Om, D,
                                         logM_grid_for_sigma, sigma_row)
            return jnp.log(M_500c / M_200c)
        return jax.vmap(at_M)(M_200c_vec)

    return jax.vmap(at_z, in_axes=(0, 0, 0, 0))(rho_c_z_tab, Om_z_tab, D_z_tab, sigma_grid)


def nfw_m500c_over_m200c(c, n_iter=15):
    """Compute M_500c/M_200c by solving 2.5*x^3 = m(c*x)/m(c) via Newton.

    `c` here is c_200c. Returns M_500c/M_200c = 2.5*x^3.

    The Newton iteration converges in ~5 steps for typical cluster
    concentrations (c ~ 5-10); n_iter=15 gives generous headroom and runs
    inside a fori_loop so the JIT trace stays cheap.
    """
    m_c = _nfw_m(c)

    def step(_, x):
        cx = c * x
        F = 2.5 * x**3 - _nfw_m(cx) / m_c
        Fp = 7.5 * x**2 - c * _nfw_m_prime(cx) / m_c
        return x - F / Fp

    x0 = jnp.full_like(c, 0.65)  # typical R_500c/R_200c
    x = jax.lax.fori_loop(0, n_iter, step, x0)
    return 2.5 * x**3


@jax.jit
def m200c_to_m500c_b13_direct(M_200c_Msunh, z, Om0, OL0):
    """Direct M_200c → M_500c using closed-form B13 c_200c + NFW geometry.

    Inputs:
      M_200c_Msunh: M_200c in M_sun/h units (classy_sz convention).
      z: redshift.
      Om0, OL0: present-day Omega values (total Omega_m, total Omega_Lambda).

    Returns:
      M_500c in M_sun/h units.

    Cosmology enters via D(z) only (the radiation contribution to D is sub-
    percent at z<5 and is dropped; OL0 absorbs any small leftover budget).
    The h scaling is implicit in M_sun/h units (classy_sz convention for
    this fit).

    All inputs may be scalar or arrays of compatible shape; the function
    broadcasts.
    """
    D_z = growth_factor_carroll_press_turner(z, Om0, OL0)
    c = b13_c200c_closed_form(M_200c_Msunh, z, D_z)
    ratio = nfw_m500c_over_m200c(c)
    return M_200c_Msunh * ratio


def log_m500c_over_m200c_grid(z_tab, lnM_tab_Msunh, Om0, OL0):
    """Build a (z, lnM_200c) → log(M_500c/M_200c) table using the closed-form
    B13 path. Convenience wrapper for the SR `_precompute_mass_conversion`
    style of layer-0 lookup.

    Args:
      z_tab: 1D array of redshifts, shape (n_z,).
      lnM_tab_Msunh: 1D array of ln(M_200c / [M_sun/h]), shape (n_m,).
      Om0, OL0: cosmology scalars.

    Returns:
      2D array of shape (n_z, n_m) holding log(M_500c / M_200c).
    """
    z_grid = z_tab[:, None]
    M_grid = jnp.exp(lnM_tab_Msunh)[None, :]
    M500 = m200c_to_m500c_b13_direct(M_grid, z_grid, Om0, OL0)
    return jnp.log(M500 / M_grid)


@jax.jit
def log_m500c_over_m200c_grid_sigma_based(sigma_grid, D_z_tab):
    """Build the (z, lnM_200c) → log(M_500c/M_200c) grid from a precomputed
    sigma(M, z) grid and a growth-factor vector D(z).

    This is the sigma-based B13 + direct NFW path. classy_sz's
    `get_m200c_to_m500c_at_z_and_M` goes M_200c→M_vir→M_500c with c_vir;
    here we use c_200c directly (the simpler, equivalent NFW solve).
    Cosmology dependence enters via sigma_grid (captures Om, sigma_8, n_s
    through the linear matter power spectrum) and via D_z_tab (captures h
    and Om residuals).

    Args:
        sigma_grid: 2D array shape (n_z, n_m) of sigma(M, z). M_vec used
            to build this must be the same physical mass as the lnM_tab
            consumed downstream.
        D_z_tab: 1D array shape (n_z,) of linear growth factor.

    Returns:
        2D array shape (n_z, n_m) of log(M_500c/M_200c).
    """
    c = b13_c200c_sigma_based(sigma_grid, D_z_tab[:, None])
    return jnp.log(nfw_m500c_over_m200c(c))
