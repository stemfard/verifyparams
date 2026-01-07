from typing import Any, Callable

from numpy import array, asarray, ndarray
from sympy import Expr, SympifyError, lambdify, sympify

from verifyparams.core._is_dtypes import is_function
from verifyparams.core.errors import SYMPIFY_ERRORS


def _symbolic_expr_err(
    fexpr: Any,
    is_variable: bool = False,
    param_name: str = "fexpr"
):
    if is_variable:
        msg = "have at least one unknown variable or expression"
    else:
        msg = "be a symbolic variable or expression"
    
    return f"Expected {param_name!r} to {msg}, got {fexpr!r}"


def sym_lambdify_expr(
    fexpr: str | Callable | list[str] | tuple[str, ...], 
    is_univariate: bool = False, 
    variables: list[str] | None = None, 
    param_name: str = 'fexpr'
) -> Callable:
    """
    Converts a symbolic equation into a NumPy-compatible function.

    Parameters
    ----------
    fexpr : {str, callable, list_like}
        The symbolic equation (or a list of symbolic equations) to be 
        converted into a Numpy-compatible function.
    is_univariate : bool, optional (default=False)
        Whether or not the equation is univariate.
    variables : array_like, optional (default=None)
        List of variable names in the equation.
    param_name : str, optional (default='fexpr')
        Name to use in error messages to describe the parameter being 
        checked.

    Returns
    -------
    callable
        A NumPy-compatible function representing the input equation.

    Examples:
    --------
    >>> import verifyparams as vp
    >>> f = vp.sym_lambdify_expr('x ** 2 + 2*x + 1')
    >>> f(2)
    9
    >>> equations = ['sin(x1) + x2 ** 2 + log(x3) - 7',
    ... '3*x1 + 2 ** x2 - x3 ** 3 + 1', 'x1 + x2 + x3 - 5']
    >>> f = vp.sym_lambdify_expr(equations)
    >>> f([4, 5/7, 3])
    array([ 8.80464777, 25.63556851,  2.71428571])
    """
    if is_function(fexpr):
        return fexpr
    elif isinstance(fexpr, (list, tuple)):
        try:
            f = sym_expr_to_numpy_function(eqtns=fexpr)
            return f
        except (ValueError, TypeError, AttributeError, SympifyError) as e:
            msg = f"Failed to convert expressions: {e}"
            raise type(e)(msg) from e
    else:
        try:
            f = sympify(a=fexpr)
        except (TypeError, ValueError, AttributeError, SympifyError) as e:
            msg = _symbolic_expr_err(
                value=fexpr, is_variable=False, param_name=param_name
            )
            raise ValueError(msg) from e
        
        if not isinstance(f, Expr):
            raise TypeError(
                f"Expected symbolic expression for {param_name!r}, "
                f"got {type(f).__name__}"
            )
            
        # univariate
        is_univariate = (
            is_univariate if isinstance(is_univariate, bool) else False
        )
        
        fvars = f.free_symbols
        nvars = len(fvars)
        
        if is_univariate and nvars != 1:
            raise ValueError(
                f"Expected a univariate expression {param_name!r}, "
                f"got expression with {nvars} variables: {fexpr!r}"
            )
            
        if variables is None:
            fvars_sorted = sorted(fvars, key=str)
        else:
            try:
                fvars_sorted = tuple(variables)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid variables specification for {param_name}: {e}"
                ) from e

        try:
            return lambdify(fvars_sorted, f, 'numpy')
        except SYMPIFY_ERRORS as e:
            raise ValueError(
                f"Failed to lambdify expression for {param_name!r}: {e}"    
            ) from e


def sym_expr_to_numpy_function(
    eqtns: list[str] | tuple[str, ...] | ndarray  
) -> Callable[[ndarray], ndarray]:
    """
    Convert a list of equations given as strings into 
    a NumPy-compatible function.

    Parameters
    ----------
    eqtns : list or tuple of str
        List of equations in string format.

    Returns
    -------
    f : callable
        Function that takes an array of variable values and returns
        an array of equation evaluations.

    Raises
    ------
    ValueError
        If any equation cannot be parsed or is invalid.
    TypeError
        If input is not a list/tuple of strings.
    """    
    if not isinstance(eqtns, (list, tuple, ndarray)):
        raise TypeError(
            f"Expected list or tuple, got {type(eqtns).__name__}"
        )
        
    if not eqtns:
        raise ValueError("Empty equation list")
    
    symbols_set = set()  # set for O(1) lookups
    sympy_eqs = []
    
    for i, eq in enumerate(eqtns):
        if not isinstance(eq, str):
            raise TypeError(
                f"Expected a string for equation at index {i}, "
                f"got {type(eq).__name__}"
            )
        
        try:
            sympy_eq = sympify(eq)
        except SYMPIFY_ERRORS as e:
            raise ValueError(
                f"{eq!r} at index {i} is invalid.\nError: {e}"
            ) from e
            
        if not isinstance(sympy_eq, Expr):
            raise TypeError(
                f"{eq!r} at index {i} did not produce valid sympy expression"
            )
        
        sympy_eqs.append(sympy_eq)
        symbols_set.update(sympy_eq.free_symbols)
        
    if not symbols_set:
        # No variables - constant functions
        symbols = ()
    else:
        # Sort symbols for consistent variable ordering
        symbols = tuple(sorted(symbols_set, key=lambda s: s.name))
    
    num_vars = len(symbols)
    
    try:
        lambda_funcs = [lambdify(symbols, eq, 'numpy') for eq in sympy_eqs]
    except SYMPIFY_ERRORS as e:
        raise ValueError(f"Failed to compile equations: {e}") from e
    
    def f(x):
        """
        Evaluate the set of equations with the given input values.

        Parameters
        ----------
        x : array-like
            Input values where each element corresponds to a variable.
            Must have length equal to number of variables.

        Returns
        -------
        arr : ndarray
            Array of equation evaluations.

        Raises
        ------
        ValueError
            If input length doesn't match number of variables.
        TypeError
            If input is not array-like.
        """
        try:
            x_arr = asarray(x, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "Expected a numeric array-like object for 'x', "
                f"got {type(x).__name__}"
            ) from e
            
        if x_arr.ndim != 1:
            raise ValueError(
                f"Expected 'x' to be 1D array, got shape {x_arr.shape}"
            )
            
        if len(x_arr) != num_vars:
            raise ValueError(
                f"Expected 'eqtns' to have {num_vars} elements, got {len(x)}"
            )
        
        if num_vars == 0:
            results = [func() for func in lambda_funcs]
        else:
            results = [func(*x_arr) for func in lambda_funcs]
        
        return array(results, dtype=float)
    
    return f