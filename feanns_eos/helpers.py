from typing import Any
import jax
from jax import numpy as jnp
from .HelmholtzModel import HelmholtzModel

from flax.training import checkpoints


def load_feanns_params(ckpt_dir: str, prefix: str='feanns_'):
    """
    Load the FE-ANN EoS parameters from a checkpoint.

    Parameters
    ----------
    ckpt_dir : str
        Checkpoint directory.
    prefix : str, optional
        Prefix of the checkpoint.

    Returns
    -------
    state_restored : dict
        Dictionary with the restored parameters.
    """
    state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=prefix)
    state_restored['features'] = list(state_restored['features'].values())

    return state_restored

# Helper jitted functions that already have the model and params bound together
# useful for density solver, phase equilibria and critical point solver
def helper_solver_funs(model: HelmholtzModel, params: Any):
    """
    Helper function to create jitted functions of the FE-ANN EoS
    These functions can be used for the density, phase equilibria and critical point solver.

    The output functions are function only alpha, rhoad and Tad. The parameters
    are already bound to the functions.

    Functions:
    helmholtz_fun: Helmholtz energy
    pressure_fun: Pressure
    dpressure_drho_fun: First derivative of the pressure with respect to density
    d2pressure_drho2_fun: Second derivative of the pressure with respect to density
    dpressure_drho_aux_fun: First derivative of the pressure with respect to density auxiliary function
    pressure_and_chempot_fun: Pressure and chemical potential
    """
    ## Helmholyz energy functions
    helmholtz_fun = jax.jit(lambda alpha, rhoad, Tad:
                            model.apply(params,
                                        jnp.atleast_1d(alpha),
                                        jnp.atleast_1d(rhoad),
                                        jnp.atleast_1d(Tad)))

    # Pressure functions
    pressure_fun = jax.jit(lambda alpha, rhoad, Tad:
                           model.pressure(params,
                                          jnp.atleast_1d(alpha),
                                          jnp.atleast_1d(rhoad),
                                          jnp.atleast_1d(Tad)))

    dpressure_drho_fun = jax.jit(lambda alpha, rhoad, Tad:
                                 model.dpressure_drho(params,
                                                      jnp.atleast_1d(alpha),
                                                      jnp.atleast_1d(rhoad),
                                                      jnp.atleast_1d(Tad)))

    d2pressure_drho2_fun = jax.jit(lambda alpha, rhoad, Tad:
                                   model.d2pressure_drho2(params,
                                                          jnp.atleast_1d(alpha),
                                                          jnp.atleast_1d(rhoad),
                                                          jnp.atleast_1d(Tad)))

    dpressure_drho_aux_fun = jax.jit(lambda rhoad, alpha, Tad:
                                     model.dpressure_drho(params,
                                                          jnp.atleast_1d(alpha),
                                                          jnp.atleast_1d(rhoad),
                                                          jnp.atleast_1d(Tad))[1])

    pressure_and_chempot_fun = jax.jit(lambda alpha, rhoad, Tad:
                                       model.pressure_and_chempot(params,
                                                                  jnp.atleast_1d(alpha),
                                                                  jnp.atleast_1d(rhoad),
                                                                  jnp.atleast_1d(Tad)))

    # compiling functions
    ones_input = (jnp.ones(1), jnp.ones(1), jnp.ones(1))
    helmholtz_fun(*ones_input)


    pressure_fun(*ones_input)
    dpressure_drho_fun(*ones_input)
    d2pressure_drho2_fun(*ones_input)
    dpressure_drho_aux_fun(*ones_input)
    pressure_and_chempot_fun(*ones_input)

    funs_dict = {'helmholtz_fun': helmholtz_fun,
                 'pressure_fun': pressure_fun,
                 'dpressure_drho_fun': dpressure_drho_fun,
                 'dpressure_drho_aux_fun': dpressure_drho_aux_fun,
                 'd2pressure_drho2_fun': d2pressure_drho2_fun,
                 'pressure_and_chempot_fun': pressure_and_chempot_fun}
    return funs_dict


# Helper jitted functions that already have the model and params bound together
# useful for density solver, phase equilibria and critical point solver
def helper_jitted_funs(model: HelmholtzModel, params: Any):
    """
    Helper function to create jitted functions of the FE-ANN EoS
    These functions can be used for the density, phase equilibria and critical point solver.

    The output functions are function only alpha, rhoad and Tad. The parameters
    are already bound to the functions.

    Functions:
    helmholtz_fun: Helmholtz energy
    dhelmholtz_drho_fun: First derivative of the Helmholtz energy with respect to density
    d2helmholtz_drho2_dT_fun: Second derivative of the Helmholtz energy with respect to density and temperature
    d2helmholtz_drho2_fun: Second derivative of the Helmholtz energy with respect to density
    d2helmholtz_fun: Second derivative of the Helmholtz energy with respect to density and temperature

    pressure_fun: Pressure
    dpressure_drho_fun: First derivative of the pressure with respect to density
    d2pressure_drho2_fun: Second derivative of the pressure with respect to density
    dpressure_drho_aux_fun: First derivative of the pressure with respect to density auxiliary function
    pressure_and_chempot_fun: Pressure and chemical potential

    chemical_potential_residual_fun: residual chemical potential function
    entropy_residual_fun: residual entropy function
    internal_energy_residual_fun: residual internal energy function
    enthalpy_residual_fun: residual Enthalpy function
    gibbs_residual_fun: residual Gibbs function

    cv_residual_fun: residual isochoric heat capacity function
    cp_residual_fun: residual isobaric heat capacity function

    thermal_expansion_coeff_fun: Thermal expansion coefficient function
    thermal_pressure_coeff_fun: Thermal pressure coefficient function
    isothermal_compressibility_fun: Isothermal compressibility function
    joule_thomson_fun: Joule-Thomson coefficient function
    thermophysical_properties_fun: Thermophysical properties function
    """

    ## Helmholyz energy functions
    helmholtz_fun = jax.jit(lambda alpha, rhoad, Tad:
                            model.apply(params,
                                        jnp.atleast_1d(alpha),
                                        jnp.atleast_1d(rhoad),
                                        jnp.atleast_1d(Tad)))

    dhelmholtz_drho_fun = jax.jit(lambda alpha, rhoad, Tad:
                                  model.apply(params,
                                              jnp.atleast_1d(alpha),
                                              jnp.atleast_1d(rhoad),
                                              jnp.atleast_1d(Tad),
                                              method=model.dhelmholtz_drho))

    d2helmholtz_drho2_dT_fun = jax.jit(lambda alpha, rhoad, Tad:
                                       model.apply(params,
                                                   jnp.atleast_1d(alpha),
                                                   jnp.atleast_1d(rhoad),
                                                   jnp.atleast_1d(Tad),
                                                   method=model.d2helmholtz_drho2_dT))

    d2helmholtz_drho2_fun = jax.jit(lambda alpha, rhoad, Tad:
                                    model.apply(params,
                                                jnp.atleast_1d(alpha),
                                                jnp.atleast_1d(rhoad),
                                                jnp.atleast_1d(Tad),
                                                method=model.d2helmholtz_drho2))

    d2helmholtz_fun = jax.jit(lambda alpha, rhoad, Tad:
                              model.apply(params,
                                          jnp.atleast_1d(alpha),
                                          jnp.atleast_1d(rhoad),
                                          jnp.atleast_1d(Tad),
                                          method=model.d2helmholtz))
    # Pressure functions
    pressure_fun = jax.jit(lambda alpha, rhoad, Tad:
                           model.pressure(params,
                                          jnp.atleast_1d(alpha),
                                          jnp.atleast_1d(rhoad),
                                          jnp.atleast_1d(Tad)))

    dpressure_drho_fun = jax.jit(lambda alpha, rhoad, Tad:
                                 model.dpressure_drho(params,
                                                      jnp.atleast_1d(alpha),
                                                      jnp.atleast_1d(rhoad),
                                                      jnp.atleast_1d(Tad)))

    d2pressure_drho2_fun = jax.jit(lambda alpha, rhoad, Tad:
                                   model.d2pressure_drho2(params,
                                                          jnp.atleast_1d(alpha),
                                                          jnp.atleast_1d(rhoad),
                                                          jnp.atleast_1d(Tad)))

    dpressure_drho_aux_fun = jax.jit(lambda rhoad, alpha, Tad:
                                     model.dpressure_drho(params,
                                                          jnp.atleast_1d(alpha),
                                                          jnp.atleast_1d(rhoad),
                                                          jnp.atleast_1d(Tad))[1])

    pressure_and_chempot_fun = jax.jit(lambda alpha, rhoad, Tad:
                                       model.pressure_and_chempot(params,
                                                                  jnp.atleast_1d(alpha),
                                                                  jnp.atleast_1d(rhoad),
                                                                  jnp.atleast_1d(Tad)))

    # first order derivatives (residual)
    chemical_potential_residual_fun = jax.jit(lambda alpha, rhoad, Tad:
                                              model.chemical_potential_residual(params,
                                                                                jnp.atleast_1d(alpha),
                                                                                jnp.atleast_1d(rhoad),
                                                                                jnp.atleast_1d(Tad)))

    entropy_residual_fun = jax.jit(lambda alpha, rhoad, Tad:
                                   model.entropy_residual(params,
                                                          jnp.atleast_1d(alpha),
                                                          jnp.atleast_1d(rhoad),
                                                          jnp.atleast_1d(Tad)))

    internal_energy_residual_fun = jax.jit(lambda alpha, rhoad, Tad:
                                           model.internal_energy_residual(params,
                                                                          jnp.atleast_1d(alpha),
                                                                          jnp.atleast_1d(rhoad),
                                                                          jnp.atleast_1d(Tad)))

    enthalpy_residual_fun = jax.jit(lambda alpha, rhoad, Tad:
                                    model.enthalpy_residual(params,
                                                            jnp.atleast_1d(alpha),
                                                            jnp.atleast_1d(rhoad),
                                                            jnp.atleast_1d(Tad)))

    gibbs_residual_fun = jax.jit(lambda alpha, rhoad, Tad:
                                 model.gibbs_residual(params,
                                                      jnp.atleast_1d(alpha),
                                                      jnp.atleast_1d(rhoad),
                                                      jnp.atleast_1d(Tad)))

    # second order derivatives (residual)
    cv_residual_fun = jax.jit(lambda alpha, rhoad, Tad:
                              model.cv_residual(params,
                                                jnp.atleast_1d(alpha),
                                                jnp.atleast_1d(rhoad),
                                                jnp.atleast_1d(Tad)))

    cp_residual_fun = jax.jit(lambda alpha, rhoad, Tad:
                              model.cp_residual(params,
                                                jnp.atleast_1d(alpha),
                                                jnp.atleast_1d(rhoad),
                                                jnp.atleast_1d(Tad)))

    # other second order derivatives (ideal+residual)
    thermal_expansion_coeff_fun = jax.jit(lambda alpha, rhoad, Tad:
                                          model.thermal_expansion_coeff(params,
                                                                        jnp.atleast_1d(alpha),
                                                                        jnp.atleast_1d(rhoad),
                                                                        jnp.atleast_1d(Tad)))

    thermal_pressure_coeff_fun = jax.jit(lambda alpha, rhoad, Tad:
                                         model.thermal_pressure_coeff(params,
                                                                      jnp.atleast_1d(alpha),
                                                                      jnp.atleast_1d(rhoad),
                                                                      jnp.atleast_1d(Tad)))

    isothermal_compressibility_fun = jax.jit(lambda alpha, rhoad, Tad:
                                             model.isothermal_compressibility(params,
                                                                              jnp.atleast_1d(alpha),
                                                                              jnp.atleast_1d(rhoad),
                                                                              jnp.atleast_1d(Tad)))

    joule_thomson_fun = jax.jit(lambda alpha, rhoad, Tad:
                                model.joule_thomson(params,
                                                    jnp.atleast_1d(alpha),
                                                    jnp.atleast_1d(rhoad),
                                                    jnp.atleast_1d(Tad)))

    thermophysical_properties_fun = jax.jit(lambda alpha, rhoad, Tad:
                                            model.thermophysical_properties(params,
                                                                            jnp.atleast_1d(alpha),
                                                                            jnp.atleast_1d(rhoad),
                                                                            jnp.atleast_1d(Tad)))

    # compiling functions
    ones_input = (jnp.ones(1), jnp.ones(1), jnp.ones(1))
    helmholtz_fun(*ones_input)
    dhelmholtz_drho_fun(*ones_input)
    d2helmholtz_drho2_dT_fun(*ones_input)
    d2helmholtz_drho2_fun(*ones_input)
    d2helmholtz_fun(*ones_input)

    pressure_fun(*ones_input)
    dpressure_drho_fun(*ones_input)
    d2pressure_drho2_fun(*ones_input)
    dpressure_drho_aux_fun(*ones_input)
    pressure_and_chempot_fun(*ones_input)

    chemical_potential_residual_fun(*ones_input)
    entropy_residual_fun(*ones_input)
    internal_energy_residual_fun(*ones_input)
    enthalpy_residual_fun(*ones_input)
    gibbs_residual_fun(*ones_input)

    cv_residual_fun(*ones_input)
    cp_residual_fun(*ones_input)

    thermal_expansion_coeff_fun(*ones_input)
    thermal_pressure_coeff_fun(*ones_input)
    isothermal_compressibility_fun(*ones_input)
    joule_thomson_fun(*ones_input)
    thermophysical_properties_fun(*ones_input)

    funs_dict = {'helmholtz_fun': helmholtz_fun,
                 'dhelmholtz_drho_fun': dhelmholtz_drho_fun,
                 'd2helmholtz_drho2_dT_fun': d2helmholtz_drho2_dT_fun,
                 'd2helmholtz_drho2_fun': d2helmholtz_drho2_fun,
                 'd2helmholtz_fun': d2helmholtz_fun,

                 'pressure_fun': pressure_fun,
                 'dpressure_drho_fun': dpressure_drho_fun,
                 'dpressure_drho_aux_fun': dpressure_drho_aux_fun,
                 'd2pressure_drho2_fun': d2pressure_drho2_fun,
                 'pressure_and_chempot_fun': pressure_and_chempot_fun,

                 'chemical_potential_residual_fun': chemical_potential_residual_fun,
                 'entropy_residual_fun': entropy_residual_fun,
                 'internal_energy_residual_fun': internal_energy_residual_fun,
                 'enthalpy_residual_fun': enthalpy_residual_fun,
                 'gibbs_residual_fun': gibbs_residual_fun,

                 'cv_residual_fun': cv_residual_fun,
                 'cp_residual_fun': cp_residual_fun,

                 'thermal_expansion_coeff_fun': thermal_expansion_coeff_fun,
                 'thermal_pressure_coeff_fun': thermal_pressure_coeff_fun,
                 'isothermal_compressibility_fun': isothermal_compressibility_fun,
                 'joule_thomson_fun': joule_thomson_fun,
                 'thermophysical_properties_fun': thermophysical_properties_fun}
    return funs_dict


# Helper function to get the alpha parameter
def helper_get_alpha(lambda_r, lambda_a):
    """
    Helper function to get the alpha parameter

    Parameters
    ----------
    lambda_r : float or array
        lambda_r parameter
    lambda_a : float or array
        lambda_a parameter

    Returns
    -------
    alpha : float or array
        alpha parameter
    """
    c_alpha = (lambda_r / (lambda_r-lambda_a)) * (lambda_r/lambda_a)**(lambda_a/(lambda_r-lambda_a))
    alpha = c_alpha*(1./(lambda_a-3) - 1./(lambda_r-3))
    return alpha
