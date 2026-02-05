from typing import Any

from numpy import (
    all, allclose, any, array_equal, asarray, diff, flatnonzero, float64, floating, floor,
    fromiter, int64, integer, isclose, issubdtype, ndarray
)
from numpy.typing import NDArray

from verifyparams.core._strings import str_data_join_contd
from verifyparams.core._decimals import numeric_format

from verifyparams.core._type_aliases import ArrayLike


def verify_numeric_arr(
    value: ndarray | list | tuple,
    n: int | None = None,
    all_integers: bool = False,
    all_positive: bool = False,
    allow_zero: bool = True,
    param_name: str = "value"
) -> ndarray:
    """
    Validate that `value` is numeric and optionally satisfies integer and
    positivity constraints.
    """
    try:
        arr = asarray(value, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Expected all values of {param_name!r} to be numeric, got {value}"
        ) from e

    # Check length
    if n is not None:
        if arr.size != n:
            raise ValueError(
                f"Expected {param_name!r} to have {n} elements, got {arr.size}"
            )

    # Check integers
    if all_integers:
        mask_invalid = arr % 1 != 0
        if any(mask_invalid):
            invalid = arr[mask_invalid]
            k = len(invalid)
            s = "" if k == 1 else "s"
            raise ValueError(
                f"Expected all values of {param_name!r} to be integers, "
                f"got {k} non-integer value{s}: "
                f"{invalid[:5]}{'...' if k > 5 else ''}"
            )
        arr = arr.astype(int64)

    # Check positivity
    if all_positive:
        if allow_zero:
            mask_invalid = arr < 0
        else:
            mask_invalid = arr <= 0
        if any(mask_invalid):
            invalid = arr[mask_invalid]
            k = len(invalid)
            s = "" if k == 1 else "s"
            raise ValueError(
                f"Expected all values of {param_name!r} to be positive, "
                f"got {k} invalid value{s}: "
                f"{invalid[:5]}{'...' if k > 5 else ''}"
            )

    return arr


def verify_data_length(
    value: ArrayLike,
    n: int,
    param_name: str = "value"
) -> ArrayLike:
    """
    Validate that `value` has exactly `n` elements.
    
    Parameters
    ----------
    value : list, tuple, or ndarray
        Input sequence.
    n : int
        Expected length of the sequence.
    param_name : str
        Name used in error messages.
        
    Returns
    -------
    value : list, tuple, or ndarray
        The original sequence if length is correct.
    """
    if not isinstance(value, (list, tuple, ndarray)):
        raise TypeError(
            f"Expected {param_name!r} to be a list, tuple, or ndarray, "
            f"got {type(value).__name__}"
        )
    
    if not isinstance(n, int):
        raise TypeError(
            f"Expected {param_name!r} length 'n' to be int, "
            f"got {type(n).__name__}"
        )

    actual_len = len(value)
    if actual_len != n:
        raise ValueError(
            f"Expected {param_name!r} to have exactly {n} "
            f"element{'s' if n != 1 else ''}, got {actual_len}"
        )

    return value


def verify_lower_lte_upper_arr(
    x: ArrayLike,
    y: ArrayLike,
    allow_equal: bool = False,
    to_array: bool = False,
    param_names: tuple[str, str] = ("x", "y")
) -> tuple[ArrayLike, ArrayLike]:
    """
    Check that all elements of x are less than (or <=) corresponding
    elements of y. Vectorized and efficient for large arrays.
    """
    param_name_x, param_name_y = param_names

    # Helper for converting input to ndarray
    def to_ndarray(arr, param_name):
        try:
            if isinstance(arr, ndarray):
                return arr.astype(float, copy=False)
            if hasattr(arr, "__len__") and not isinstance(arr, (str, bytes)):
                return asarray(arr, dtype=float)
            return fromiter(arr, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot convert {param_name!r} to numeric array: {e}"
            ) from e

    arr1 = to_ndarray(x, param_name_x)
    arr2 = to_ndarray(y, param_name_y)

    # Vectorized check
    if allow_equal:
        mask_invalid = arr1 > arr2
    else:
        mask_invalid = arr1 >= arr2

    if any(mask_invalid):
        invalid_pairs = list(zip(arr1[mask_invalid], arr2[mask_invalid]))
        n = len(invalid_pairs)
        s = "" if n == 1 else "s"
        raise ValueError(
            f"Expected corresponding values of {param_name_x!r} to be less "
            f"than those of {param_name_y!r}, got {invalid_pairs} invalid pair{s}"
        )

    if to_array:
        return arr1, arr2
    return x, y


def verify_elements_in_range(
    data: list[int | float] | NDArray,
    lower: int | float = 0,
    upper: int | float = 4,
    par_name: str = "data"
) -> list[int | float]:
    """
    Verify that all elements in the sequence are within the specified 
    range.
    
    Parameters
    ----------
    data : list[int | float]
        List of numerical elements to check.
    lower : int | float, optional (default=0)
        The lower bound of the range (inclusive).
    upper : int | float, optional (default=4)
        The upper bound of the range (inclusive).
    par_name : str, optional (default="data")
        Name of the input parameter for error messages.
    
    Returns
    -------
    data : list | tuple | NDArray
        The input sequence if all elements are within range.
    
    Raises
    ------
    ValueError
        If any elements are outside the specified range.
    
    Examples
    --------
    >>> verify_elements_in_range([1, 2, 3], 0, 5)
    [1, 2, 3]
    
    >>> verify_elements_in_range([1, 2, 6], 0, 5, "scores")
    ValueError: Expected all values of 'scores' to be between 0 and 5 inclusive, 
    got 6
    
    >>> verify_elements_in_range([1.0, 2.0, 3.0], 0, 4)
    [1, 2, 3]
    """
    arr = asarray(elements_outside)
    elements_outside = arr[(arr < lower) | (arr > upper)]
    
    if elements_outside:
        lower_fmt = numeric_format(lower)
        upper_fmt = numeric_format(upper)
        outside_fmt = numeric_format(elements_outside)
        
        if len(elements_outside) == 1:
            elements_str = f"value {outside_fmt[0]}"
        else:
            elements_str = f"values {str_data_join_contd(outside_fmt)}"
        
        raise ValueError(
            f"Expected all values of {par_name!r} to be between "
            f"{lower_fmt} and {upper_fmt} inclusive, got {elements_str}"
        )
    
    return data


def verify_elements_in_range(
    data: ArrayLike,
    lower: float,
    upper: float,
    param_name: str = "data"
) -> ndarray:
    """
    Verify that all elements in the sequence are within the specified range.
    """
    arr = asarray(data, dtype=float)
    
    mask_outside = (arr < lower) | (arr > upper)
    if any(mask_outside):
        elements_outside = arr[mask_outside]
        elements_str = ", ".join(str(x) for x in elements_outside[:5])
        if len(elements_outside) > 5:
            elements_str += f", ... (+{len(elements_outside)-5} more)"
        
        raise ValueError(
            f"Expected all values of {param_name!r} to be between {lower} "
            f"and {upper} inclusive, got {elements_str}"
        )
    
    return arr


def verify_all_integers(
    value: Any,
    allow_float_ints: bool = True,
    param_name: str = "values"
) -> NDArray:
    """
    Vectorized integer validation.

    Parameters
    ----------
    value : Any
        Input sequence to validate (list, tuple, or ndarray).
    allow_float_ints : bool
        If True, allows floats like 1.0, 2.0 as integers.
    param_name : str
        Parameter name for error messages.

    Returns
    -------
    ndarray[int]
        Array of integers.
    """
    try:
        arr = asarray(value)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name!r} to be list, tuple, or ndarray, "
            f"got {type(value).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e

    if not issubdtype(arr.dtype, integer):
        if allow_float_ints:
            mask_valid = (arr == floor(arr))
            if not all(mask_valid):
                indices = flatnonzero(~mask_valid)
                examples = ", ".join(f"index {i}: {arr[i]}" for i in indices[:5])
                if len(indices) > 5:
                    examples += f", ..., {len(indices)-5} more"
                raise ValueError(
                    f"Expected all values of {param_name!r} to be integers, "
                    f"got {len(indices)} non-integers (e.g., {examples})"
                )
            arr = arr.astype(int64)
        else:
            raise ValueError(
                f"Expected all values of {param_name!r} to be integers, "
                f"got dtype {arr.dtype}"
            )

    return arr


def verify_all_positive(
    data: ArrayLike,
    include_zero: bool = False,
    to_array: bool = False,
    param_name: str = 'data'
) -> ndarray:
    """
    Validate that all values in `data` are positive.

    Parameters
    ----------
    data : list, tuple, or ndarray
        Input sequence.
    include_zero : bool
        If True, zero is allowed.
    to_array : bool
        If True, always return a NumPy array.
    param_name : str
        Name used in error messages.

    Returns
    -------
    arr : ndarray
        Validated numeric array.
    """
    if isinstance(data, ndarray):
        arr = data.astype(float64, copy=False)
    else:
        try:
            if hasattr(data, '__len__') and not isinstance(data, (str, bytes)):
                arr = asarray(data, dtype=float64)
            else:
                arr = fromiter(data, dtype=float64)
        except Exception as e:
            raise TypeError(
                f"Cannot convert {param_name!r} to numeric array: {e}"
            ) from e

    mask_invalid = arr < 0 if include_zero else arr <= 0
    if any(mask_invalid):
        invalid_values = arr[mask_invalid]
        n_invalid = len(invalid_values)
        s = "" if n_invalid == 1 else "s"
        examples = ", ".join(str(x) for x in invalid_values[:5])
        if n_invalid > 5:
            examples += f", ... (+{n_invalid - 5} more)"
        raise ValueError(
            f"Expected all values of {param_name!r} to be positive, "
            f"got {n_invalid} invalid value{s}: {examples}"
        )

    return arr if to_array else data
    

class LengthMismatchError(ValueError):
    """
    Raised when multiple arrays do not have the same length.

    This error provides detailed information about which arrays
    have mismatched lengths and what the expected length was.
    """
    
    def __init__(
        self,
        lengths: list[int],
        expected_length: int,
        param_names: list[str]
    ):
        """
        Parameters
        ----------
        lengths : list[int]
            Actual lengths of each array.
        expected_length : int
            The length that all arrays were expected to have.
        param_names : list[str]
            Names of the parameters corresponding to the arrays.
        """
        self.param_names = param_names
        self.lengths = lengths
        self.expected_length = expected_length
        
        # Build error message
        mismatched = []
        for name, length in zip(param_names, lengths):
            if length != expected_length:
                mismatched.append(f"{name} (len={length})")
        
        if mismatched:
            message = (
                f"Arrays must have equal length. "
                f"Expected length {expected_length}. "
                f"Mismatched arrays: {', '.join(mismatched)}"
            )
        else:
            message = (
                f"Expected all arrays to have exactly {expected_length} "
                "elements"
            )
        
        super().__init__(message)


def verify_len_equal(
    *arrays: Any,
    param_names: list[str] | None = None
) -> tuple[Any, ...]:
    """
    Verify that all input arrays have equal length.
    """
    n_arrays = len(arrays)
    if n_arrays <= 1:
        return arrays

    if param_names is None:
        param_names = [f"array{i+1}" for i in range(n_arrays)]
    elif len(param_names) != n_arrays:
        raise ValueError(
            f"Expected {n_arrays} parameter names, got {len(param_names)}"
        )

    lengths = []
    for name, arr in zip(param_names, arrays):
        try:
            lengths.append(len(arr))
        except TypeError as e:
            raise TypeError(
                f"{name!r} has no length (type {type(arr).__name__})"
            ) from e

    expected_length = lengths[0]
    mismatched = [
        f"{name} (len={length})" for name, length in zip(param_names, lengths)
        if length != expected_length
    ]
    
    if mismatched:
        raise LengthMismatchError(
            param_names=param_names,
            lengths=lengths,
            expected_length=expected_length
        )

    return arrays


def verify_diff_constant(
    value: ArrayLike,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    to_array: bool = False,
    param_name: str = 'x'
) -> ArrayLike:
    """
    Verify that differences between consecutive elements are constant.
    """
    # Determine input type for return
    if isinstance(value, list):
        val_type = "list"
    elif isinstance(value, tuple):
        val_type = "tuple"
    elif isinstance(value, ndarray):
        val_type = "array"
    else:
        raise TypeError(
            f"Expected {param_name!r} to be numeric array-like, "
            f"got {type(value).__name__}"
        )

    # Convert to float64 array
    if isinstance(value, ndarray):
        arr = value.astype(float64, copy=False)
    else:
        try:
            arr = asarray(value, dtype=float64)
        except Exception as e:
            raise TypeError(
                f"Expected {param_name!r} to be numeric array-like, "
                f"got {type(value).__name__}"
            ) from e

    # Nothing to check if fewer than 2 elements
    if arr.size < 2:
        return arr if to_array or val_type == "array" else arr.tolist()

    # Vectorized difference
    diffs = diff(arr)

    # Check for constant difference
    if not allclose(diffs, diffs[0], rtol=rtol, atol=atol):
        # Find first mismatch
        for i, diff_val in enumerate(diffs[1:], start=1):
            if not isclose(diff_val, diffs[0], rtol=rtol, atol=atol):
                raise ValueError(
                    f"Differences in {param_name!r} are not constant. "
                    f"diff[{i}] = {diff_val:.12g}, expected ≈ {diffs[0]:.12g}"
                )

    # Return in original type
    if to_array or val_type == "array":
        return arr
    elif val_type == "list":
        return arr.tolist()
    else:
        return tuple(arr.tolist())


def verify_strictly_increasing(
    value: Any,
    param_name: str = "x"
) -> NDArray[float64]:
    """
    Fast strict increasing check using NumPy.
    """
    try:
        arr = asarray(value, dtype=float64)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name!r} to be numeric array-like, "
            f"got {type(value).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e

    if arr.size < 2:
        return arr

    diffs = diff(arr)
    mask = diffs <= 0
    if any(mask):
        first_violation = flatnonzero(mask)[0]  # avoids recomputing mask
        raise ValueError(
            f"Expected {param_name!r} to be strictly increasing. "
            f"Element {first_violation + 1} ({arr[first_violation + 1]:g}) <= "
            f"element {first_violation} ({arr[first_violation]:g})"
        )

    return arr


def verify_distinct(
    x: ArrayLike,
    y: ArrayLike,
    tolerance: float = 1e-12,
    check_length: bool = True,
    param_names: tuple[str, str] = ('x', 'y')
) -> tuple[ArrayLike, ArrayLike]:
    """
    Verify that two arrays/lists/tuples are distinct.
    """
    param_name_x, param_name_y = (
        param_names if isinstance(param_names, (tuple, list)) else ('x', 'y')
    )

    # Type validation
    for arr, name in ((x, param_name_x), (y, param_name_y)):
        if not isinstance(arr, (list, tuple, ndarray)):
            raise TypeError(
                f"Expected {name!r} to be list, tuple, or ndarray, "
                f"got {type(arr).__name__}"
            )

    # Check length
    if check_length and len(x) != len(y):
        raise ValueError(
            f"Expected both arrays to have the same length, "
            f"got {param_name_x!r} (len={len(x)}) and {param_name_y!r} "
            f"(len={len(y)})"
        )

    # Fast path for different lengths
    if len(x) != len(y):
        return x, y  # automatically distinct

    # Fast path for NumPy numeric arrays
    if isinstance(x, ndarray) and isinstance(y, ndarray):
        try:
            if allclose(x, y, rtol=tolerance, atol=tolerance):
                raise ValueError(
                    f"Expected arrays {param_name_x!r} and {param_name_y!r} "
                    f"to be distinct, got identical arrays (within tolerance "
                    f"{tolerance})"
                )
        except (TypeError, ValueError) as e:
            if array_equal(x, y):
                raise ValueError(
                    f"Expected arrays {param_name_x!r} and {param_name_y!r} "
                    f"to be distinct"
                ) from e
        return x, y

    # Fallback: elementwise comparison for lists/tuples
    for xi, yi in zip(x, y):
        if xi != yi:
            return x, y

    # All elements identical
    raise ValueError(
        f"Expected arrays {param_name_x!r} and {param_name_y!r} to be "
        f"distinct, got identical arrays"
    )
    

def verify_not_constant(
    value,
    tolerance: float = 1e-12,
    param_name: str = "x"
):
    """
    Verify that an array is not constant (all values not the same).
    """
    try:
        arr = asarray(value, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Expected {param_name!r} to be numeric array-like, "
            f"got {type(value).__name__}"
        ) from e

    # Scalars and empty arrays are constant
    if arr.ndim == 0 or arr.size < 2:
        raise ValueError(
            f"All values of {param_name!r} are the same ({arr.item():g})"
        )

    first = arr[0]

    if issubdtype(arr.dtype, floating):
        if allclose(arr, first, rtol=tolerance, atol=tolerance):
            raise ValueError(
                f"All values of {param_name!r} are approximately equal "
                f"(≈{first:g} within tolerance {tolerance})"
            )
    else:
        if all(arr == first):
            raise ValueError(
                f"All values of {param_name!r} are the same ({first!r})"
            )

    # Return original container type
    if isinstance(value, ndarray):
        return arr
    if isinstance(value, list):
        return arr.tolist()
    if isinstance(value, tuple):
        return tuple(arr.tolist())
    return arr