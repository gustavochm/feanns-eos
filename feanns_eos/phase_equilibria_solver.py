from typing import Sequence, Callable
import jax.numpy as jnp
from .density_solver import density_solver
from scipy.optimize import root
import numpy as np


def of_critical_point(inc: Sequence[float], alpha: float, d2pressure_drho2_fun: Callable):
    """
    Objective function for the critical point.

    Parameters
    ----------
    inc : Sequence[float]
        Independent variables [rhoad, Tad].
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    d2pressure_drho2_fun : Callable
        Function that returns the reduced pressure and its first and second derivatives

    Returns
    -------
    of : jnp.ndarray
        Objective function for the critical point [d2P_dV, d2P_dV2]
    """

    rhoad, Tad = inc
    alpha = jnp.atleast_1d(alpha) 
    rhoad = jnp.atleast_1d(rhoad)
    Tad = jnp.atleast_1d(Tad)

    out = d2pressure_drho2_fun(alpha, rhoad, Tad)
    Pad, dP_drho, d2P_drho2 = out

    dP_dV = dP_drho
    d2P_dV2 = 2. * dP_drho + rhoad * d2P_drho2

    of = jnp.hstack([dP_dV, d2P_dV2])
    return of


def critical_point_solver(alpha, fun_dic,
                          inc0=[0.3, 1.3], root_kwargs: dict={},
                          full_output: bool=False):
    """
    Critical point solver for the FE-ANN EoS.

    Parameters
    ----------
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    fun_dic : dict
        Dictionary with the functions to solve the density, pressure and chemical potential.
        Must include the functions: pressure_fun, d2pressure_drho2_fun.
    inc0 : Sequence[float]
        Initial guess for the critical point [rhoad0, Tad0]. The default is [0.3, 1.3].
    root_kwargs : dict, optional
        Keyword arguments for the root solver. The default is {}.
    full_output : bool, optional
        If True, returns a dictionary with the critical point. The default is False.

    Returns
    -------
    rhocad : float
        Critical reduced density.
    Tcad : float
        Critical reduced temperature.
    Pcad : float
        Critical reduced pressure.
    """
    pressure_fun = fun_dic['pressure_fun']
    d2pressure_drho2_fun = fun_dic['d2pressure_drho2_fun']

    sol = root(of_critical_point, inc0, args=(alpha, d2pressure_drho2_fun), **root_kwargs)
    rhocad, Tcad = sol.x

    Pcad = float(sol.fun[0])
    Pcad = np.asarray(pressure_fun(alpha, rhocad, Tcad))[0]

    if full_output:
        out = {'rhocad': rhocad, 'Tcad': Tcad, 'Pcad': Pcad, 'success': sol.success}
    else:
        out = rhocad, Tcad, Pcad
    return out


def of_triple_point(inc0: Sequence[float], alpha: float, pressure_and_chempot_fun: Callable):
    """
    Objective function for triple point.

    Parameters
    ----------
    inc00 : Sequence[float]
        Initial guesses for the densities and triple temperature [rho0_1, rho0_2, rho0_3, T0].
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    pressure_and_chempot_fun : Callable
        Function that returns the pressure and chemical potential.

    Returns
    -------
    of : jnp.ndarray
        Objective function for the VLE solver [P_1 - P_2, P_1 - P_3, mu_1 - mu_2, mu_1 - mu_3]
    """
    inc0 = jnp.asarray(inc0).flatten()
    T = inc0[3]
    rho0 = inc0[:3]
    alpha = jnp.array([alpha, alpha, alpha]).flatten()
    Tad = jnp.array([T, T, T]).flatten()

    pressure, chem_pot = pressure_and_chempot_fun(alpha, rho0, Tad)
    of = jnp.array([pressure[0]-pressure[1], 
                    pressure[0]-pressure[2], 
                    chem_pot[0]-chem_pot[1], 
                    chem_pot[0]-chem_pot[2]])

    return of


def triple_point_solver(alpha, fun_dic,
                        inc0=[1e-3, 0.85, 1.0, 0.68], root_kwargs: dict={},
                        full_output: bool=False):
    """
    Triple point solver for the FE-ANN EoS.

    Parameters
    ----------
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    fun_dic : dict
        Dictionary with the functions to solve the density, pressure and chemical potential.
        Must include the functions: pressure_fun, pressure_and_chempot_fun.
    inc0 : Sequence[float], optional
        Initial guesses for the densities and triple temperature [rho0_1, rho0_2, rho0_3, T0].
        The default is [1e-3, 0.85, 1.0, 0.68].
    root_kwargs : dict, optional
        Keyword arguments for the root solver. The default is {}.
    full_output : bool, optional
        If True, returns a dictionary with the triple point. The default is False.

    Returns
    -------
    rhovad_triple : float
        Vapour density at the triple point.
    rholad_triple : float
        Liquid density at the triple point.
    rhosad_triple : float
        Solid density at the triple point.
    T_triple : float
        Triple temperature.
    P_triple : float   
        Triple pressure.
    """
    pressure_fun = fun_dic['pressure_fun']
    pressure_and_chempot_fun = fun_dic['pressure_and_chempot_fun']

    sol_triple = root(of_triple_point, inc0, args=(alpha, pressure_and_chempot_fun))
    rhovad_triple = sol_triple.x[0]
    rholad_triple = sol_triple.x[1]
    rhosad_triple = sol_triple.x[2]
    T_triple = sol_triple.x[3]
    P_triple = float(pressure_fun(alpha, rhovad_triple, T_triple))

    if full_output:
        out = {'rhovad': rhovad_triple, 'rholad': rholad_triple, 
               'rhosad': rhosad_triple, 'Tad': T_triple, 'Pad': P_triple, 'success': sol_triple.success}
    else:
        out = rhovad_triple, rholad_triple, rhosad_triple, T_triple, P_triple

    return out


def of_two_phase(rho0: Sequence[float], alpha: float, T: float, pressure_and_chempot_fun: Callable):
    """
    Objective function for the two phase equilibria solver.

    Parameters
    ----------
    rho0 : Sequence[float]
        Initial guesses for the densities [rho0_1, rho0_2].
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    T : float
        Temperature.
    pressure_and_chempot_fun : Callable
        Function that returns the pressure and chemical potential.

    Returns
    -------
    of : jnp.ndarray
        Objective function for the VLE solver [P_1 - P_2, mu_1 - mu_2]
    """

    rho0 = jnp.asarray(rho0).flatten()
    alpha = jnp.array([alpha, alpha]).flatten()
    Tad = jnp.array([T, T]).flatten()

    pressure, chem_pot = pressure_and_chempot_fun(alpha, rho0, Tad)
    of = jnp.hstack([jnp.diff(pressure), jnp.diff(chem_pot)])
    return of


def vle_solver(alpha, Tad, fun_dic, Pad0=None, critical=None, rho0: Sequence=[None, None], 
               max_iter: int=10, tol: float=1e-8, good_initial: bool=False):
    """
    Vapour-liquid Equilibria solver for the FE-ANN EoS.

    If good_initial is False, the VLE is solved using the isofugacity method.
    If good_initial is True, the VLE is solved using muldimensional system of equations [dP, dmu].

    Parameters
    ----------
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    Tad : float
        Temperature.
    fun_dic : dict
        Dictionary with the functions to solve the density, pressure and chemical potential.
    Pad0 : float, optional
        Initial guess for the pressure. The default is None.
    critical : tuple, optional
        Critical point (rhocad, Tcad, Pcad). The default is None. If None, the critical point is solved.
    rho0 : Sequence, optional
        Initial guess for the densities [rho0_vap, rho0_liq]. The default is [None, None].
    max_iter : int, optional
        Maximum number of iterations. The default is 10.
    tol : float, optional
        Tolerance for the convergence. The default is 1e-8.
    good_initial : bool, optional
        If True, the VLE is solved using muldimensional system of equations [dP, dmu]. The default is False.
        Often this method works better close to the critical point.

    Returns
    -------
    P : float
        Pressure.
    density_vap : float
        Vapour density.
    density_liq : float
        Liquid density.

    """
    helmholtz_fun = fun_dic['helmholtz_fun']
    pressure_fun = fun_dic['pressure_fun']
    d2pressure_drho2_fun = fun_dic['d2pressure_drho2_fun']
    pressure_and_chempot_fun = fun_dic['pressure_and_chempot_fun']

    # Solving critical if not provided:
    if critical is None:
        inc0 = [0.3, 1.3]
        sol = root(of_critical_point, inc0, args=(alpha, d2pressure_drho2_fun))
        rhocad, Tcad = sol.x
        Pcad = float(pressure_fun(alpha, rhocad, Tcad))
    else:
        if isinstance(critical, dict):
            rhocad = critical['rhocad']
            Tcad = critical['Tcad']
            Pcad = critical['Pcad']
        else:
            rhocad, Tcad, Pcad = critical

    # if temperature than critical temperature there is no equilibria
    if Tad > Tcad:
        return Pcad, rhocad, rhocad

    # getting initial guess for density
    density_vap = rho0[0]
    density_liq = rho0[1]

    if Pad0 is None:
        P0 = 0.0
        rholiq0, Pcheck0 = density_solver(alpha, Tad, P0, state='L',
                                          density_solver_fun_dic=fun_dic,
                                          rho_min=rhocad)
        AresP0 = helmholtz_fun(alpha, rholiq0, Tad)
        fugP0 = float(Tad * rholiq0 * jnp.exp(AresP0/Tad - 1.))
        Pad0 = fugP0
        density_liq = rholiq0

    # solving phase equilibria otherwise
    alpha_fun = jnp.array([alpha, alpha]).flatten()
    Tad_fun = jnp.array([Tad, Tad]).flatten()

    if good_initial:
        rho0 = jnp.array([density_vap, density_liq]).flatten()
        sol_vle = root(of_two_phase, rho0, args=(alpha, Tad, pressure_and_chempot_fun))
        density_vap, density_liq = sol_vle.x
        Pad = float(pressure_fun(alpha, density_vap, Tad))
        success = sol_vle.success
        return Pad, density_vap, density_liq

    # First iteration
    Pad = Pad0
    density_liq, P_check_liq = density_solver(alpha, Tad, Pad, state='L', density_solver_fun_dic=fun_dic,
                                              rho0=density_liq)
    density_vap, P_check_vap = density_solver(alpha, Tad, Pad, state='V', density_solver_fun_dic=fun_dic,
                                              rho0=density_vap)

    density_fun = jnp.array([density_vap, density_liq]).flatten()

    Ares = helmholtz_fun(alpha_fun, density_fun, Tad_fun)
    Z = Pad / (density_fun * Tad_fun)
    lnfug = Ares/Tad + (Z - 1.) - jnp.log(Z)
    OF = lnfug[0] - lnfug[1]
    dOF = (1./density_vap - 1./density_liq) / Tad
    dPad = OF / dOF
    if dPad > Pad:
        dPad = dPad / 2
    Pad -= float(dPad)

    for i in range(max_iter):
        density_liq, P_check_liq = density_solver(alpha, Tad, Pad, state='L', 
                                                  density_solver_fun_dic=fun_dic, rho0=density_liq)
        density_vap, P_check_vap = density_solver(alpha, Tad, Pad, state='V',
                                                  density_solver_fun_dic=fun_dic, rho0=density_vap)

        density_fun = jnp.array([density_vap, density_liq]).flatten()

        Ares = helmholtz_fun(alpha_fun, density_fun, Tad_fun)
        Z = Pad / (density_fun * Tad_fun)
        lnfug = Ares/Tad + (Z - 1.) - jnp.log(Z)
        OF = lnfug[0] - lnfug[1]
        dOF = (1./density_vap - 1./density_liq) / Tad
        dPad = OF / dOF
        if dPad > Pad:
            dPad = dPad / 2
        Pad -= float(dPad)

        success = abs(OF) <= tol
        # print(i, Pad, OF, density_vap, density_liq)
        if success:
            break

    if not success:
        rho0 = jnp.array([density_vap, density_liq]).flatten()
        sol_vle = root(of_two_phase, rho0, args=(alpha, Tad, pressure_and_chempot_fun))
        density_vap, density_liq = sol_vle.x
        Pad = float(pressure_fun(alpha, density_vap, Tad))
        success = sol_vle.success

    if success:
        density_vap = float(density_vap)
        density_liq = float(density_liq)
    else:
        density_vap = np.nan
        density_liq = np.nan
        Pad = np.nan

    return Pad, density_vap, density_liq


def sle_solver(alpha, Tad, fun_dic, rho0: Sequence=[None, None],
               root_kwargs: dict={}):

    pressure_fun = fun_dic['pressure_fun']
    pressure_and_chempot_fun = fun_dic['pressure_and_chempot_fun']

    # getting initial guess for density
    density_liq = rho0[0]
    density_sol = rho0[1]

    rho0 = jnp.array([density_liq, density_sol]).flatten()
    sol_sle = root(of_two_phase, rho0, args=(alpha, Tad, pressure_and_chempot_fun), **root_kwargs)
    density_liq, density_sol = sol_sle.x
    Pad = float(pressure_fun(alpha, density_liq, Tad))
    success = sol_sle.success

    if not success:
        density_liq = np.nan
        density_sol = np.nan
        Pad = np.nan

    return Pad, density_liq, density_sol

sve_solver = sle_solver
