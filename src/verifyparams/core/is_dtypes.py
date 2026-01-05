from sympy import Expr, SympifyError, sympify


def is_function(obj):
    """

    Check if the given object is a function.

    Parameters
    ----------
    obj: object
        The object to be checked.

    Returns
    -------
    bool: 
        True if the object is a function, False otherwise.

    """
    import numpy as np
    try:
        if callable(obj) and (
            isinstance(obj, type(lambda x: x)) or hasattr(np, obj.__name__)):
            return True
        else:
            return False
    except Exception as e:
        raise TypeError(e)
    
    
def is_symexpr(fexpr: str | Expr) -> bool:
    """
    Check if expression is symbolic (i.e. contains unknown variables).

    Parameters
    ----------
    fexpr : {str, sympy.Expr}
        An object representing the value to be tested.

    Returns
    -------
    result: bool
        True/False

    Examples
    --------
    >>> import verifyparams as vpa

    >>> stm.is_symexpr('pi/4')
    False

    >>> stm.is_symexpr('pi/x')
    True

    >>> g = 'x^2 + x * y - 5'
    >>> stm.is_symexpr(g)
    True
    """
    try:
        result = len(sympify(str(fexpr)).free_symbols) > 0
    except (TypeError, ValueError, AttributeError, SympifyError):
        result = False

    return result