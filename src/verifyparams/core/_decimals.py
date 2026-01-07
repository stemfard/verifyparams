from typing import Any
import numbers

from numpy import (
    any, all, dtype, empty, empty_like, float64, floating, integer,
    issubdtype, ndarray, ndindex
)

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def numeric_format(value: Any) -> Any:
    """
    Recursively normalize numeric values by converting whole-valued
    floats to integers while preserving container structure.
    
    Parameters
    ----------
    value : Any
        A numeric value, or a nested container (list, tuple, or 
        numpy.ndarray) containing numeric values.
    
    Returns
    -------
    Any
        Formatted value with whole floats converted to integers.
    
    Raises
    ------
    TypeError
        If input is not numeric or a supported container.
    
    Examples
    --------
    >>> numeric_format(3.0)
    3
    >>> numeric_format([1.0, 2.5, 3.0])
    [1, 2.5, 3]
    >>> numeric_format(array([1.0, 2.5, 3.0]))
    array([1, 2.5, 3], dtype=object)
    """
    # Handle numpy arrays (with Numba optimization for float64)
    if isinstance(value, ndarray):
        return _format_array(value)
    
    # Handle Python lists
    if isinstance(value, list):
        return [numeric_format(v) for v in value]
    
    # Handle Python tuples
    if isinstance(value, tuple):
        return tuple(numeric_format(v) for v in value)
    
    # Handle Python float
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    
    # Handle Python int
    if isinstance(value, int):
        return value
    
    # Handle numpy scalar floats
    if isinstance(value, floating):
        fval = float(value)
        return int(fval) if fval.is_integer() else value
    
    # Handle numpy scalar ints
    if isinstance(value, integer):
        return int(value)
    
    # Handle other numeric types
    if isinstance(value, numbers.Number):
        return value
    
    # Unsupported type
    raise TypeError(
        f"numeric_format() only supports numeric values, lists, tuples, "
        f"and NumPy arrays; got {type(value).__name__}"
    )


def _format_array(arr: ndarray) -> ndarray:
    """Format numpy array with Numba optimization when possible."""
    # Integer arrays need no formatting
    if issubdtype(arr.dtype, integer):
        return arr
    
    # Float arrays
    if issubdtype(arr.dtype, floating):
        # Try Numba for float64 if available
        if HAS_NUMBA and arr.dtype == float64:
            try:
                return _numba_format_float64(arr)
            except Exception:
                pass  # Fall back to Python
        
        return _format_float_array_python(arr)
    
    # Object arrays
    if arr.dtype == dtype('O'):
        return _format_object_array(arr)
    
    # Other dtypes
    return arr


def _format_float_array_python(arr: ndarray) -> ndarray:
    """Python implementation for float arrays."""
    # Find whole numbers
    remainder = arr % 1.0
    mask = remainder == 0.0
    
    # Check if any conversion needed
    if not any(mask):
        return arr
    
    # All whole numbers - convert entire array to int
    if all(mask):
        return arr.astype(int)
    
    # Mixed whole/non-whole - use object dtype
    result = empty(arr.shape, dtype=object)
    result[mask] = arr[mask].astype(int)
    result[~mask] = arr[~mask]
    return result


def _format_object_array(arr: ndarray) -> ndarray:
    """Format object arrays recursively."""
    # Vectorized but preserves object dtype
    result = empty_like(arr, dtype=object)
    for idx in ndindex(arr.shape):
        result[idx] = numeric_format(arr[idx])
    return result


# Numba-accelerated function (only defined if numba is available)
if HAS_NUMBA:
    @numba.njit(cache=True)
    def _numba_format_float64(arr: ndarray) -> ndarray:
        """Numba formatter that properly creates object arrays."""
        # Must use object array to store mixed ints/floats
        result = empty(arr.shape, dtype=dtype('O'))
        for i in range(arr.size):
            val = arr.flat[i]
            int_val = int(val)
            if val == int_val:
                result.flat[i] = int_val  # Store as Python int
            else:
                result.flat[i] = val      # Store as numpy float64
        return result
else:
    _numba_format_float64 = None