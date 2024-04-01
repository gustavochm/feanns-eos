import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar, brentq


def density_newton_lim(alpha, Tad, Pad_set, dpressure_drho_fun,
                       rhoad0=None, rhoad_lower=0.0, rhoad_upper=1.2,
                       eps=1e-8, tol=1e-8, max_iter=10):
    """
    Density solver using bounded Newton-Raphson method. To be used with the
    FE-ANN EoS model.

    Parameters
    ----------
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    Tad : float
        Reduced temperature.
    Pad_set : float
        Set reduced pressure.
    dpressure_drho_fun : function
        Function that returns the reduced pressure and its derivative with
        respect to density.
    rhoad0 : float, optional
        Initial guess for the reduced density. The default is None.
    rhoad_lower : float, optional
        Lower bound for the reduced density. The default is 0.0.
    rhoad_upper : float, optional
        Upper bound for the reduced density. The default is 1.2.
    eps : float, optional
        Small number to avoid division by zero. The default is 1e-8.
    tol : float, optional
        Tolerance for the density solver. The default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations for the density solver. The default is 10.

    Returns
    -------
    rhoad : float
        Reduced density.
    Pad_model : float
        Model reduced pressure.
    """
    alpha = jnp.atleast_1d(alpha)
    Tad = jnp.atleast_1d(Tad)
    Pad_set = jnp.atleast_1d(Pad_set)

    rhoad_lower = jnp.atleast_1d(rhoad_lower)
    rhoad_upper = jnp.atleast_1d(rhoad_upper)

    if rhoad0 is None:
        rhoad = (rhoad_lower + rhoad_upper) / 2
    else: 
        rhoad = jnp.atleast_1d(rhoad0)
    rhoad = jnp.atleast_1d(rhoad)
    Pad_model, dPad_model = dpressure_drho_fun(alpha, rhoad, Tad)
    # print(-1, rhoad, Pad_model, Pad_set)

    for i in range(max_iter):
        rhoad_old = rhoad
        OF = Pad_model - Pad_set
        dOF = dPad_model
        drhoad = OF / dOF
        rhoad_new = rhoad - drhoad

        if OF > 0:
            rhoad_upper = rhoad + eps
        else:
            rhoad_lower = rhoad - eps

        if rhoad_lower < rhoad_new < rhoad_upper:
            rhoad = rhoad_new
        else:
            rhoad = (rhoad_lower + rhoad_upper) / 2

        if jnp.abs(rhoad - rhoad_old) < tol: break

        Pad_model, dPad_model = dpressure_drho_fun(alpha, rhoad, Tad)

        # print(i, rhoad, rhoad_lower , rhoad_upper, Pad_model, Pad_set, OF, drhoad)

    return rhoad, Pad_model


def density_topliss(alpha, Tad, Pad, state, density_solver_fun_dic,
                    eps=1e-8, tol=1e-8, max_iter=10, rho_max=1.2):
    """
    Density solver using Topliss method. To be used with the FE-ANN EoS model.

    Parameters
    ----------
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    Tad : float
        Reduced temperature.
    Pad : float
        Set reduced pressure.
    state : str
        State of the fluid. 'L' for liquid and 'V' for vapour.
    density_solver_fun_dic : dict
        Dictionary with the functions to calculate the reduced pressure and its
        derivatives with respect to density.
    eps : float, optional
        Small number to avoid division by zero. The default is 1e-8.
    tol : float, optional
        Tolerance for the density solver. The default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations for the density solver. The default is 10.
    rho_max : float, optional
        Maximum reduced density. The default is 1.2.

    Returns
    -------
    rhoad : float
        Reduced density.
    Pad_model : float
        Model reduced pressure.
    """

    state = state.upper()

    pressure_fun = density_solver_fun_dic['pressure_fun']
    dpressure_drho_fun = density_solver_fun_dic['dpressure_drho_fun']
    d2pressure_drho2_fun = density_solver_fun_dic['d2pressure_drho2_fun']
    dpressure_drho_aux_fun = density_solver_fun_dic['dpressure_drho_aux_fun']

    Pad_set = Pad

    alpha = jnp.atleast_1d(alpha)
    Tad = jnp.atleast_1d(Tad)
    Pad_set = jnp.atleast_1d(Pad_set)

    # upper boundary limit at infinity pressure
    etamax = 0.7405
    rho_lim = (6 * etamax) / np.pi
    ub_sucess = False
    rho_ub = np.array([0.4 * rho_lim])
    it = 0
    Pad_ub_model, dPad_ub_model = dpressure_drho_fun(alpha, rho_ub, Tad)
    while not ub_sucess and it < 5:
        if rho_ub > rho_max:
            rho_ub = jnp.array([rho_max])
        it += 1
        Pad_ub_model, dPad_ub_model = dpressure_drho_fun(alpha, rho_ub, Tad)
        rho_ub += 0.1 * rho_lim
        ub_sucess = Pad_ub_model > Pad_set and dPad_ub_model > 0
        if rho_ub > rho_max:
            rho_ub = jnp.array([rho_max])

    # lower boundary a zero density
    rho_lb = jnp.array([0.0])
    Pad_lb_model, dPad_lb_model, d2Pad_lb_model = d2pressure_drho2_fun(alpha, rho_lb, Tad)
    if d2Pad_lb_model > 0:
        flag = 3
    else:
        flag = 1

    # Stage 1
    bracket = [rho_lb, rho_ub]
    sol_inf = minimize_scalar(dpressure_drho_aux_fun, args=(alpha, Tad),
                              bounds=bracket,  method='Bounded', options={'xatol': 1e-3})
    rho_inf = jnp.atleast_1d(sol_inf.x)
    dP_inf = sol_inf.fun
    if dP_inf > 0:
        flag = 3
    else:
        flag = 2

    # Stage 2
    if flag == 2:
        if state == 'L':
            bracket[0] = rho_inf
        elif state == 'V':
            bracket[1] = rho_inf
        rho_ext = brentq(dpressure_drho_aux_fun, bracket[0], bracket[1],
                         args=(alpha, Tad), xtol=1e-3)
        rho_ext = np.atleast_1d(rho_ext)
        Pad_ext_model = pressure_fun(alpha, rho_ext, Tad)
        if Pad_ext_model > Pad_set and state == 'V':
            bracket[1] = rho_ext
        elif Pad_ext_model < Pad_set and state == 'L':
            bracket[0] = rho_ext
        else:
            flag = -1

    if flag == -1:
        rhoad, Pad_model = jnp.nan, jnp.nan
    else:
        rhoad, Pad_model = density_newton_lim(alpha, Tad, Pad_set, dpressure_drho_fun,
                                   rhoad0=None, rhoad_lower=bracket[0], rhoad_upper=bracket[1],
                                   eps=eps, tol=tol, max_iter=max_iter)
    rhoad = jnp.atleast_1d(rhoad)
    Pad_model = jnp.atleast_1d(Pad_model)
    return rhoad, Pad_model


def density_solver(alpha, Tad, Pad, state, density_solver_fun_dic, 
                   rho0=None, rho_min=0.0, rho_max=1.2,
                   eps=1e-8, tol=1e-8, max_iter=15):

    """
    Density solver to be used with the FE-ANN EoS model.

    Parameters
    ----------
        Parameters
    ----------
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    Tad : float
        Reduced temperature.
    Pad : float
        Set reduced pressure.
    state : str
        State of the fluid. 'L' for liquid and 'V' for vapour.
    density_solver_fun_dic : dict
        Dictionary with the functions to calculate the reduced pressure and its
        derivatives with respect to density.
    rho0 : float, optional
        Initial guess for the reduced density. The default is None.
    rho_min : float, optional
        Lower bound for the reduced density. The default is 0.0.
    rho_max : float, optional
        Upper bound for the reduced density. The default is 1.2.
    eps : float, optional
        Small number to avoid division by zero. The default is 1e-8.
    tol : float, optional
        Tolerance for the density solver. The default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations for the density solver. The default is 15.

    Returns
    -------
    rhoad : float
        Reduced density.
    Pad_model : float
        Model reduced pressure.
    """
    if rho0 is None:
        rhoad, Pad_model = density_topliss(alpha, Tad, Pad, state, density_solver_fun_dic,
                                           rho_max=rho_max, eps=eps, tol=tol, max_iter=max_iter)
    else:
        rhoad, Pad_model = density_newton_lim(alpha, Tad, Pad, density_solver_fun_dic['dpressure_drho_fun'],
                                              rhoad0=rho0, rhoad_lower=rho_min, rhoad_upper=rho_max,
                                              eps=eps, tol=tol, max_iter=max_iter)

    return rhoad, Pad_model
