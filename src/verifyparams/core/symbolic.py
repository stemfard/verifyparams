from typing import Any, Callable

from numpy import array
from sympy import SympifyError, lambdify, sympify

from verifyparams.core.is_dtypes import is_function, is_symexpr


def _symbolic_expr_err(
    value: Any,
    is_variable: bool = False,
    param_name: str = "expr"
):
    if is_variable:
        msg = "have at least one unknown variable or expression"
    else:
        msg = "be a symbolic variable or expression"
    
    return f"Expected {param_name!r} to {msg}, got {value!r}"


def sym_lambdify_expr(
    fexpr: str | Callable | list[str], 
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
    >>> import verifyparams as vpa
    >>> f = stm.sym_lambdify_expr('x ** 2 + 2*x + 1')
    >>> f(2)
    9
    >>> equations = ['sin(x1) + x2 ** 2 + log(x3) - 7',
    ... '3*x1 + 2 ** x2 - x3 ** 3 + 1', 'x1 + x2 + x3 - 5']
    >>> f = stm.sym_lambdify_expr(equations)
    >>> f([4, 5/7, 3])
    array([ 8.80464777, 25.63556851,  2.71428571])
    """
    
    if is_function(fexpr):
        return fexpr
    elif isinstance(fexpr, (list, tuple)):
        f = sym_expr_to_numpy_function(fexpr)
        return f
    else:
        try:
            f = sympify(expr_array=fexpr)
        except (TypeError, ValueError, AttributeError, SympifyError) as e:
            msg = _symbolic_expr_err(
                value=fexpr, is_variable=False, param_name=param_name
            )
            raise type(e)(msg) from e
        
        # univariate
        is_univariate = (
            is_univariate if isinstance(is_univariate, bool) else False
        )

        # f - continued
        if is_symexpr(f):
            fvars = f.free_symbols
            nvars = len(fvars)
            if is_univariate and nvars != 1:
                raise ValueError(
                    f"Expected {param_name!r} to be a univariate "
                    f"polynomial, got {fexpr!r}"
                )
        else:
            msg = _symbolic_expr_err(
                value=fexpr, is_variable=False, param_name=param_name
            )
            raise type(str(e))(msg) from e
        
        if variables is None:
            fvars = tuple(fvars)
        else: 
            try:
                fvars = tuple(variables)
            except (TypeError, ValueError) as e:
                raise type(e)(str(e)) from e
            
        fexpr = lambdify(fvars, f, 'numpy')

    return fexpr


def sym_expr_to_numpy_function(equations: list[str]) -> Callable:
    """
    Convert a list of equations given as strings into 
    a NumPy-compatible function.

    Parameters
    ----------
    equations: list, of str
        List of equations in string format.

    Returns
    -------
    f: A Numpy function.
    """
    symbols = []
    sympy_eqs = []
    
    for eq in equations:
        sympy_eq = sympify(eq)
        sympy_eqs.append(sympy_eq)
        for symbol in sympy_eq.free_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
    
    num_vars = len(symbols)
    lambda_funcs = [lambdify(symbols, eq, 'numpy') for eq in sympy_eqs]
    
    def f(x):
        """
        Evaluate the set of equations with the given input values.

        Parameters
        ----------
        x: {list, tuple}
            Input array where each element corresponds to a variable.

        Returns
        -------
        numpy_array: Results of the equations.
        """
        # Ensure the input array has the correct length
        if len(x) != num_vars:
            raise ValueError(
                f"Expected input array to have {num_vars} elements, "
                f"got {len(x)}"
            )
        
        # Evaluate each lambda function with the input values
        results = [f(*x) for f in lambda_funcs]
        return array(results)
    
    return f