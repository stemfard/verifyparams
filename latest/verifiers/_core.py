from typing import Any, Callable
from numpy import array, asarray, ndarray
from sympy import Expr, SympifyError, lambdify, sympify


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


def is_maths_function(obj: Any) -> bool:
    """
    Check whether an object is a usable mathematical function.

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
        True if `obj` is callable and not a class, False otherwise.

    Notes
    -----
    Classes are callable but are excluded since they represent
    constructors rather than mathematical functions. Callable
    instances, including NumPy ufuncs such as ``np.sin``, are accepted.

    Examples
    --------
    >>> import numpy as np
    >>> import stemcore as stc
    >>> 
    >>> stc.is_maths_function(np.sin)
    True
    >>> stc.is_maths_function(lambda x: x**2)
    True
    >>> stc.is_maths_function(int)
    False
    """
    if not callable(obj):
        return False

    # Classes are callable but represent constructors
    if isinstance(obj, type):
        return False

    return True


def is_symexpr(obj: Any) -> bool:
    """
    Check whether an object is a symbolic expression with free symbols.

    A symbolic expression is defined as one containing at least one
    free symbol (variable), excluding defined constants.

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
        True if `obj` represents a symbolic expression with free 
        symbols, False otherwise.

    Notes
    -----
    Only SymPy expressions are considered. String inputs are parsed using
    ``sympify`` and rejected if parsing fails or if no free symbols are
    present.

    Examples
    --------
    >>> import sympy as sym
    >>> x, y = sym.symbols('x y')

    >>> is_symexpr(sym.pi/4)
    False
    >>> is_symexpr(sym.pi/x)
    True
    >>> is_symexpr(x**2 + x*y - 5)
    True
    >>> is_symexpr('x**2 + y')
    True
    >>> is_symexpr('3.14')
    False
    >>> is_symexpr(42)
    False
    """
    if isinstance(obj, Expr):
        return bool(obj.free_symbols)

    if isinstance(obj, str):
        try:
            expr = sympify(obj)
        except (TypeError, ValueError, AttributeError, SympifyError):
            return False

        if isinstance(expr, Expr):
            return bool(expr.free_symbols)

    return False


def _sym_expr_to_numpy_function(
    eqtns: list[str] | tuple[str, ...] | ndarray
) -> Callable[[ndarray], ndarray]:
    """
    Convert a list of symbolic equations (as strings) into 
    a NumPy-compatible function.

    This function parses each equation string into a SymPy expression,
    identifies all free symbols (variables), and returns a callable
    that evaluates all equations given a numeric array of variable
    values.

    Parameters
    ----------
    eqtns : list, tuple, or ndarray of str
        List of equations in string format. Each string should be
        a valid SymPy-parsable expression.

    Returns
    -------
    f : callable
        Function that takes a 1D numeric array of length equal to the
        number of variables in all equations and returns a NumPy array
        of evaluations.
        - Input `x` must correspond in order to the sorted symbols.
        - Output is a 1D `ndarray` with each element representing the
          evaluation of one equation.

    Raises
    ------
    TypeError
        If `eqtns` is not a list, tuple, or ndarray, or if any element
        is not a string.
    ValueError
        If `eqtns` is empty, contains invalid equations, or cannot be
        compiled.
    
    Notes
    -----
    - Symbols are sorted by name to ensure consistent ordering.
    - Equations without free symbols are treated as constant functions.
    - The returned function accepts numeric array-like input and
      evaluates all equations simultaneously.
    - The function uses SymPy's ``lambdify`` with the ``'numpy'``
      backend for fast numeric evaluation.

    Examples
    --------
    >>> import numpy as np
    >>> import stemcore as stc
    >>>
    >>> # 2 equations as strings
    >>> f = stc._sym_expr_to_numpy_function(["x + y", "x**2 - z"])
    >>> f(np.array([1, 2, 3]))  # x=1, y=2, z=3
    array([3., -2.])
    
    >>> # Constant equation
    >>> g = stc._sym_expr_to_numpy_function(["5"])
    >>> g(np.array([]))
    array([5.])
    
    See Also
    --------
    sym_lambdify_expr : Convert a single symbolic expression or list of
        expressions into a NumPy-compatible function.
    is_symexpr : Check whether an object is a symbolic expression with
        free symbols.
    
    """
    if not isinstance(eqtns, (list, tuple, ndarray)):
        raise TypeError(
            f"Expected 'eqtns' to be a list or tuple, "
            f"got {type(eqtns).__name__}"
        )

    if not eqtns:
        raise ValueError("Empty equation list")

    symbols_set = set()
    sympy_eqs = []

    for i, eq in enumerate(eqtns):
        if not isinstance(eq, str):
            raise TypeError(
                f"Expected equation at index {i} to be a string, "
                f"got {type(eq).__name__}"
            )

        try:
            sympy_eq = sympify(eq)
        except (TypeError, ValueError, AttributeError, SympifyError) as e:
            raise ValueError(
                f"{eq!r} at index {i} is invalid.\nError: {e}"
            ) from e

        if not isinstance(sympy_eq, Expr):
            raise TypeError(
                f"{eq!r} at index {i} did not produce a valid sympy expression"
            )

        sympy_eqs.append(sympy_eq)
        symbols_set.update(sympy_eq.free_symbols)

    if not symbols_set:
        symbols = ()
    else:
        symbols = tuple(sorted(symbols_set, key=lambda s: s.name))

    num_vars = len(symbols)

    try:
        lambda_funcs = [lambdify(symbols, eq, 'numpy') for eq in sympy_eqs]
    except (TypeError, ValueError, AttributeError, SympifyError) as e:
        raise ValueError(f"Failed to compile equations: {e}") from e

    def f(x):
        """
        Evaluate the set of equations with the given numeric input.

        Parameters
        ----------
        x : array-like
            1D array-like input where each element corresponds to
            a variable. Length must match the number of variables.

        Returns
        -------
        arr : ndarray
            Evaluated results of all equations as a NumPy array.

        Raises
        ------
        TypeError
            If input is not array-like or cannot be converted to float.
        ValueError
            If input length does not match the number of variables.
        """
        try:
            x_arr = asarray(x, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "Expected 'x' to be a numeric array-like object, "
                f"got {type(x).__name__}"
            ) from e

        if x_arr.ndim != 1:
            raise ValueError(
                f"Expected 'x' to be a 1D array, got shape {x_arr.shape}"
            )

        if len(x_arr) != num_vars:
            s = "" if num_vars == 1 else "s"
            raise ValueError(
                f"Expected 'x' to have {num_vars} equation{s}, got {len(x_arr)}"
            )

        if num_vars == 0:
            results = [func() for func in lambda_funcs]
        else:
            results = [func(*x_arr) for func in lambda_funcs]

        return array(results, dtype=float)

    return f


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
    >>> import stemcore as stc
    >>> f = stc.sym_lambdify_expr('x ** 2 + 2*x + 1')
    >>> f(2)
    9
    >>> equations = ['sin(x1) + x2 ** 2 + log(x3) - 7',
    ... '3*x1 + 2 ** x2 - x3 ** 3 + 1', 'x1 + x2 + x3 - 5']
    >>> f = stc.sym_lambdify_expr(equations)
    >>> f([4, 5/7, 3])
    array([ -6.14798613, -12.35932929,   2.71428571])
    """
    if is_maths_function(fexpr):
        return fexpr
    elif isinstance(fexpr, (list, tuple)):
        try:
            f = _sym_expr_to_numpy_function(eqtns=fexpr)
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
                f"Expected {param_name!r} to be a symbolic expression, "
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
                f"Expected {param_name!r} to be a univariate expression, "
                f"got expression with {nvars} variables: {fexpr!r}"
            )
            
        if variables is None:
            fvars_sorted = sorted(fvars, key=str)
        else:
            try:
                fvars_sorted = tuple(variables)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid variables specification for {param_name!r}: {e}"
                ) from e

        try:
            return lambdify(fvars_sorted, f, 'numpy')
        except (TypeError, ValueError, AttributeError, SympifyError) as e:
            raise ValueError(
                f"Failed to lambdify expression for {param_name!r}: {e}"    
            ) from e