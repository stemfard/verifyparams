from typing import Any

from numpy import ndarray

def format_num(value: Any) -> Any:
    """
    Convert floats that are whole numbers to ints.
    Works recursively on lists, tuples, and NumPy arrays.

    Parameters
    ----------
    value : int, float, list, tuple, ndarray
        Number or iterable of numbers.

    Returns
    -------
    int, float, or list/tuple of int/float
        Numbers with floats converted to ints if whole.
    """
    # Handle lists recursively
    if isinstance(value, list):
        return [format_num(v) for v in value]
    
    # Handle tuples recursively
    if isinstance(value, tuple):
        return tuple(format_num(v) for v in value)
    
    # Handle NumPy arrays recursively
    if isinstance(value, ndarray):
        return [format_num(v) for v in value.flatten()]

    # Single number
    if isinstance(value, float) and value.is_integer():
        return int(value)
    
    return value