from typing import Any, Callable, Sequence, Union
from functools import partial
from jax import vmap
from jax.tree_util import tree_map, tree_transpose, tree_structure
from jax._src.api_util import check_callable, argnums_partial, _ensure_index
from jax._src.util import wraps
from jax._src import linear_util as lu
from jax._src.api import (_vjp, _check_input_dtype_jacrev, _check_output_dtype_jacrev,
                          _std_basis, _jacrev_unravel, 
                          _jvp, _check_input_dtype_jacfwd, _check_output_dtype_jacfwd,
                          _jacfwd_unravel)


def val_and_jacrev(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                   has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
    """Value and Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

    Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
    then a pair of (jacobian, auxiliary_data) is returned.

    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> def f(x):
    ...   return jnp.asarray(
    ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
    ...
    >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))
    [[ 1.       0.       0.     ]
    [ 0.       0.       5.     ]
    [ 0.      16.      -2.     ]
    [ 1.6209   0.       0.84147]]
    """
    check_callable(fun)

    docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

    @wraps(fun, docstr=docstr, argnums=argnums)
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
        jac = vmap(pullback)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
        jac_tree = tree_transpose(tree_structure(example_args), tree_structure(y), jac_tree)
        if not has_aux:
            return jac_tree, y
        else:
            return jac_tree, (y, aux)

    return jacfun


def val_and_jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           has_aux: bool = False, holomorphic: bool = False) -> Callable:
    """Val and Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

    Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation. If ``has_aux`` is True
    then a pair of (jacobian, auxiliary_data) is returned.

    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> def f(x):
    ...   return jnp.asarray(
    ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
    ...
    >>> print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))
    [[ 1.       0.       0.     ]
    [ 0.       0.       5.     ]
    [ 0.      16.      -2.     ]
    [ 1.6209   0.       0.84147]]
    """
    check_callable(fun)
    argnums = _ensure_index(argnums)

    docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

    @wraps(fun, docstr=docstr, argnums=argnums)
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        else:
            pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
        tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if not has_aux:
            return jac_tree, y
        else:
            return jac_tree, (y, aux)

    return jacfun
