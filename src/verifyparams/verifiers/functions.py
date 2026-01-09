from typing import Any, Callable

from sympy import SympifyError

from verifyparams.verifiers._core import sym_lambdify_expr


def verify_function(
    f: Any,
    is_univariate: bool = False,
    variables: list[str] | None = None,
    param_name: str = "f"
) -> Callable:
    """
    Validate and convert to callable function.
    
    Parameters
    ----------
    f : Any
        Input to validate/convert.
    is_univariate : bool
        Whether function should be univariate.
    variables : list of str, optional
        Variable names for symbolic expressions.
    param_name : str
        Parameter name for errors.
    
    Returns
    -------
    Callable
        Validated/callable function.
    """    
    try:
        return sym_lambdify_expr(
            fexpr=f,
            is_univariate=is_univariate,
            variables=variables,
            par_name=param_name
        )
    except (ValueError, TypeError, AttributeError, SympifyError) as e:
        if callable(f):
            raise ValueError(
                f"Function {param_name!r} is callable but failed "
                f"validation: {e}"
            ) from e
        else:
            raise TypeError(
                f"Expected {param_name!r} to be callable or convertible "
                f"expression, got {type(f).__name__}: {f!r}"
            ) from e