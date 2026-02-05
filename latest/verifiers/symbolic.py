from typing import Callable

from sympy import Expr


def verify_symbolic_expr(
    fexpr: str | Expr | Callable, nvars: int | None = None    
) -> Expr:
    """Check if symbolic expression"""
    return fexpr