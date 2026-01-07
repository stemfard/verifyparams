from typing import Any

from sympy import Expr, sympify

from verifyparams.core.errors import SYMPIFY_ERRORS


def is_maths_function(obj: Any) -> bool:
    """
    Check if object is already a usable mathematical function.
    """
    if not callable(obj):
        return False
    
    # Reject classes (they're constructors, not functions)
    # numpy ufuncs like np.sin are instances, not the ufunc class
    if isinstance(obj, type):
        return False
    
    return True


def is_symexpr(obj: Any) -> bool:
    """
    Check if an object is a symbolic expression (contains variables).
    
    A symbolic expression is defined as having at least one free symbol
    (variable) that is not a defined constant.
    
    Parameters
    ----------
    obj : Any
        Object to check. Can be:
        - sympy.Expr object
        - String representation of symbolic expression
        - Any other object (will return `False` if not symbolic)
    
    Returns
    -------
    bool
        `True` if the object represents a symbolic expression with 
        variables, `False` otherwise.
    
    Examples
    --------
    >>> import sympy as sym
    >>> x, y = sym.symbols('x y')
    
    >>> is_symexpr(sym.pi/4)      # Constant expression
    False
    
    >>> is_symexpr(sym.pi/x)      # Contains variable x
    True
    
    >>> is_symexpr(x**2 + x*y - 5)  # Multiple variables
    True
    
    >>> is_symexpr('x**2 + y')   # String input
    True
    
    >>> is_symexpr('3.14')       # Numeric string
    False
    
    >>> is_symexpr(42)           # Plain number
    False
    
    >>> is_symexpr([x, y])       # List (not expression)
    False
    """
    if isinstance(obj, Expr):
        return len(obj.free_symbols) > 0
    
    if isinstance(obj, str):
        try:
            expr = sympify(obj)
            if isinstance(expr, Expr):
                return len(expr.free_symbols) > 0
            else:
                return False
        except SYMPIFY_ERRORS:
            return False

    return False