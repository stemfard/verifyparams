from typing import Any

from numpy import (
    all, allclose, array, array_equal, asarray, float64, floating, isclose,
    issubdtype, ndarray, diff, any, where
)

from verifyparams.core.decimals import format_num


def verify_elements_in_range(
    lst: list[int | float],
    lower: int | float = 0,
    upper: int | float = 4,
    par_name: str = "input"
) -> list[int | float]:
    """
    Verify that all elements in the list are within the specified range.
    
    Parameters
    ----------
    lst : list[int | float]
        List of numerical elements to check.
    lower : int | float, optional (default=0)
        The lower bound of the range (inclusive).
    upper : int | float, optional (default=4)
        The upper bound of the range (inclusive).
    par_name : str, optional (default="input")
        Name of the input parameter for error messages.
    
    Returns
    -------
    list[int | float]
        The input list if all elements are within range.
    
    Raises
    ------
    ValueError
        If any elements are outside the specified range.
    
    Examples
    --------
    >>> verify_elements_in_range([1, 2, 3], 0, 5)
    [1, 2, 3]
    
    >>> verify_elements_in_range([1, 2, 6], 0, 5, "scores")
    ValueError: Expected all values of 'scores' to be between 0 and 5 inclusive; 
    got 6
    
    >>> verify_elements_in_range([1.0, 2.0, 3.0], 0, 4)
    [1, 2, 3]
    """
    
    # Find elements outside the range
    elements_outside = [x for x in lst if x < lower or x > upper]
    
    if elements_outside:
        # Format bounds and elements for error message
        lower_fmt = format_num(lower)
        upper_fmt = format_num(upper)
        
        # Format the out-of-range elements
        outside_fmt = format_num(elements_outside)
        
        # Create informative error message
        if len(elements_outside) == 1:
            elements_str = f"value {outside_fmt[0]}"
        else:
            elements_str = f"values {', '.join(map(str, outside_fmt))}"
        
        raise ValueError(
            f"Expected all values of {par_name!r} to be between "
            f"{lower_fmt} and {upper_fmt} inclusive; got {elements_str}"
        )
    
    # Return the formatted list for consistency
    return format_num(lst)


def verify_all_integers(
    value: Any,
    param_name: str = "values",
    allow_float_ints: bool = True
) -> list[int]:
    """
    Fast integer validation.
    
    Parameters
    ----------
    value : Any
        Input sequence to validate.
    param_name : str
        Parameter name for error messages.
    allow_float_ints : bool
        If True, allows floats like 1.0, 2.0.
    
    Returns
    -------
    List[int]
        List of integers.
    """
    result = []
    non_integers = []
    
    # Handle different input types
    if isinstance(value, (list, tuple)):
        iterable = enumerate(value)
    else:
        try:
            iterable = enumerate(value)
        except TypeError:
            raise TypeError(
                f"Expected {param_name!r} to be list, tuple or ndarray; "
                f"got {type(value).__name__}"
            )
    
    for i, value in iterable:
        if isinstance(value, int):
            result.append(value)
        elif isinstance(value, float):
            if allow_float_ints and value.is_integer():
                result.append(int(value))
            else:
                non_integers.append((i, value))
        else:
            # Try to convert
            try:
                int_val = int(value)
                result.append(int_val)
            except (ValueError, TypeError):
                non_integers.append((i, value))
    
    # Check for errors
    n = len(non_integers)
    if non_integers:
        if n == 1:
            idx, val = non_integers[0]
            raise ValueError(
                f"Expected all values of {param_name!r} to be integers; "
                f"got a non-integer at index {idx}: {val!r}"
            )
        else:
            k = 5
            examples = (
                ", ".join(f"index {i}: {v!r}" for i, v in non_integers[:k])
            )
            if n > k:
                examples += f", ..., {n - k} more"
            
            raise ValueError(
                f"Expected all values of {param_name!r} to be integers, "
                f"got {len(non_integers)} non-integers (e.g., {examples})"
            )
    
    return result


def verify_all_positive(
    user_input: Any,
    param_name: str = 'input',
    include_zero: bool = False,
    to_array: bool = False
) -> list[float]:
    """
    Simple and fast positive validation.
    
    Parameters
    ----------
    user_input : Any
        Input to validate.
    param_name : str
        Parameter name.
    include_zero : bool
        Whether to allow zero.
    
    Returns
    -------
    List[float]
        Validated values.
    """
    if isinstance(user_input, (list, tuple)):
        values = user_input
    else:
        # Try to iterate
        try:
            values = list(user_input)
        except TypeError as e:
            raise TypeError(
                f"Expected {param_name!r} to be iterable; "
                f"got {type(user_input).__name__}"
            ) from e
    
    # Single-pass validation and conversion
    result = []
    errors = []
    
    for i, val in enumerate(values):
        try:
            # Convert to number
            if isinstance(val, str):
                num = float(val)
                if num % 1 == 0:
                    num = int(num)
            elif isinstance(val, float):
                num = float(val)
            elif isinstance(val, int):
                num = int(val)
            else:
                num = float(val)  # it will fail if not convertible
                if num % 1 == 0:
                    num = int(num)
        except (ValueError, TypeError):
            errors.append((i, val))
            continue
        
        # Check positivity
        if include_zero:
            if num < 0:
                errors.append((i, num))
            else:
                result.append(num)
        else:
            if num <= 0:
                errors.append((i, num))
            else:
                result.append(num)
    
    # Handle errors
    if errors:
        if len(errors) == 1:
            idx, val = errors[0]
            raise ValueError(
                f"Expected all values of {param_name!r} to be "
                f"{'non-negative' if include_zero else 'positive'}, "
                f"got invalid value at index {idx}: {val}"
            )
        else:
            error_examples = ", ".join(
                f"index {idx}: {val}" for idx, val in errors[:3]
            )
            if len(errors) > 3:
                error_examples += f" ... and {len(errors) - 3} more"
            
            raise ValueError(
                f"Expected all values of {param_name!r} to be "
                f"{'non-negative' if include_zero else 'positive'}, "
                f"got {len(errors)} invalid values: {error_examples}"
            )
            
    if to_array:
        result = array(result)
    
    return result


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

    Parameters
    ----------
    *arrays : Any
        Arrays or sequence-like objects whose lengths are to be compared.
    param_names : list[str] | None, optional
        Names of the parameters corresponding to each array.
        If not provided, default names ('array1', 'array2', ...) are used.

    Returns
    -------
    tuple[Any, ...]
        The input arrays unchanged, if all lengths are equal.

    Raises
    ------
    TypeError
        If any input does not support `len()`.
    ValueError
        If `param_names` length does not match the number of arrays.
    LengthMismatchError
        If the arrays do not all have the same length.
    """
    n_arrays = len(arrays)
    
    if n_arrays <= 1:
        return arrays
    
    # Setup parameter names
    if param_names is None:
        param_names = [f"array{i + 1}" for i in range(n_arrays)]
    
    if len(param_names) != n_arrays:
        raise ValueError(
            f"`param_names` length ({len(param_names)}) must match "
            f"arrays count ({n_arrays})"
        )
    
    # Get lengths
    lengths = []
    for i, arr in enumerate(arrays):
        try:
            lengths.append(len(arr))
        except TypeError as e:
            raise TypeError(
                f"{param_names[i]!r} has no length (type: {type(arr).__name__})"
            ) from e
    
    # Check equality
    expected_length = lengths[0]
    all_equal = all(length == expected_length for length in lengths)
    
    if not all_equal:
        raise LengthMismatchError(
            param_names=param_names,
            lengths=lengths,
            expected_length=expected_length
        )
    
    return arrays


def verify_diff_constant(
    user_input: Any,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    to_array: bool = False,
    param_name: str = 'x'
) -> list[int | float] | tuple[int | float] | ndarray:
    """
    Verify that differences between consecutive elements are constant.

    This function checks whether the input sequence forms an arithmetic
    progression within a given numerical tolerance.

    Parameters
    ----------
    user_input : Any
        Input sequence (list, tuple, or NumPy array) containing 
        numeric values.
    rtol : float, optional
        Relative tolerance for difference comparison (default is 1e-9).
    atol : float, optional
        Absolute tolerance for difference comparison (default is 1e-12).
    to_array : bool, optional
        If `True`, always return a NumPy array regardless of input type.
        If `False`, preserve the input container type when possible.
    param_name : str, optional
        Name of the parameter for error messages.

    Returns
    -------
    result : list[int | float] | tuple[int | float] | ndarray
        The validated input sequence, with its original type preserved
        unless `to_array=True`.

    Raises
    ------
    TypeError
        If `user_input` is not array-like or contains non-numeric values.
    ValueError
        If differences between consecutive elements are not constant.
    """
    msg = (
        f"Expected {param_name!r} to be numeric array-like; "
        f"got {type(user_input).__name__}"
    )
    
    if isinstance(user_input, list):
        val_type = "list"
    elif isinstance(user_input, tuple):
        val_type = "tuple"
    else:
        if to_array or isinstance(user_input, ndarray):
            val_type = "array"
        else:
            raise TypeError(msg)
    
    if not isinstance(user_input, ndarray):
        try:
            arr = asarray(user_input, dtype=float64)
        except TypeError as e:
            raise TypeError(msg) from e
        except ValueError as e:
            raise ValueError(str(e)) from e
    
    if arr.size < 2:
        return arr.tolist()
    
    # Calculate differences (vectorized, very fast)
    diffs = diff(arr)
    
    # Check if all differences are equal within tolerance
    if not allclose(diffs, diffs[0], rtol=rtol, atol=atol):
        # Find first non-matching difference for error message
        first_diff = diffs[0]
        for i, diff in enumerate(diffs[1:], 1):
            if not isclose(diff, first_diff, rtol=rtol, atol=atol):
                raise ValueError(
                    f"Differences in {param_name!r} are not constant. "
                    f"diff[{i}] = {diff:.6g}, expected ≈ {first_diff:.6g}"
                )
                
    if val_type == "list":
        result = arr.tolist()
    elif val_type == "tuple":
        result = tuple(arr.tolist())
    else:
        result = arr
    
    return result


def verify_strictly_increasing(
    user_input: Any,
    param_name: str = "x"
) -> list[int | float] | tuple[int | float] | ndarray:
    """
    Fast strict increasing check using numpy.
    
    Parameters
    ----------
    user_input : array_like
        Input sequence.
    param_name : str
        Parameter name.
    
    Returns
    -------
    arr : ndarray
        Validated array.
    """
    try:
        arr = asarray(user_input, dtype=float64)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name!r} to be numeric array-like; "
            f"got {type(user_input).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e
    
    if arr.size < 2:
        return arr
    
    # Check if strictly increasing
    diffs = diff(arr)
    if any(diffs <= 0):
        # Find first violation
        violations = where(diffs <= 0)[0]
        first_violation = violations[0]
        
        raise ValueError(
            f"Expected {param_name!r} to be strictly increasing. "
            f"Element {first_violation + 1} ({arr[first_violation + 1]:g}) <= "
            f"element {first_violation} ({arr[first_violation]:g})"
        )
    
    return arr


def verify_distinct(
    x: list | tuple | ndarray,
    y: list | tuple | ndarray,
    tolerance: float = 1e-12,
    param_name_x: str = 'x',
    param_name_y: str = 'y',
    check_length: bool = True
) -> tuple[list | tuple | ndarray, list | tuple | ndarray]:
    """
    Distinct array verification.
    """
    # Type validation
    if not isinstance(x, (list, tuple, ndarray)):
        raise TypeError(
            f"Expected {param_name_x!r} to be a list, tuple, or ndarray; "
            f"got {type(x).__name__}"
        )
    
    if not isinstance(y, (list, tuple, ndarray)):
        raise TypeError(
            f"Expected {param_name_y!r} to be a list, tuple, or ndarray; "
            f"got {type(y).__name__}"
        )
    
    # Length check
    if check_length and len(x) != len(y):
        raise ValueError(
            f"Expected both arrays to have the same length; "
            f"got {param_name_x!r} (len={len(x)}) and "
            f"{param_name_y!r} (len={len(y)})"
        )
    
    # if different lengths, then they can't be identical
    if len(x) != len(y):
        return x, y
    
    # Fast path for numpy arrays
    if isinstance(x, ndarray) and isinstance(y, ndarray):
        try:
            # Try numeric comparison with tolerance
            if allclose(x, y, rtol=tolerance, atol=tolerance):
                raise ValueError(
                    f"Expected arrays {param_name_x!r} and {param_name_y!r} "
                    "to be distinct; got identical arrays (within tolerance "
                    f"{tolerance})"
                )
        except (TypeError, ValueError): # Non-numeric arrays will faile
            if array_equal(x, y): # so use exact equality
                raise ValueError(
                    f"Expected arrays {param_name_x!r} and {param_name_y!r} "
                    f"to be distinct; got identical arrays"
                )
        return x, y
    
    # compare element by element with early exit
    for xi, yi in zip(x, y):
        if xi != yi: # Found a difference, so arrays are distinct
            return x, y
    
    # all elements were equal
    raise ValueError(
        f"Expected arrays {param_name_x!r} and {param_name_y!r} to be "
        f"distinct; got identical arrays"
    )
    

def verify_not_constant(
    user_input: Any,
    param_name: str = "x",
    tolerance: float = 1e-12
) -> list | ndarray:
    """
    Verify that an array is not constant (all values not the same).
    
    Parameters
    ----------
    user_input : array_like
        Input to check.
    param_name : str
        Parameter name for error messages.
    tolerance : float
        Tolerance for floating point comparisons.
    
    Returns
    -------
    list or ndarray
        Validated input in appropriate format.
    
    Raises
    ------
    TypeError
        If input cannot be converted to array.
    ValueError
        If all values are (approximately) equal.
    
    Examples
    --------
    >>> verify_not_constant([1, 2, 3])
    [1, 2, 3]
    
    >>> verify_not_constant([1, 1, 1])
    ValueError: All values of 'values' are the same (1)
    
    >>> verify_not_constant([1.0, 1.0000000001])
    [1.0, 1.0000000001]  # Different within tolerance
    
    >>> verify_not_constant([1.0, 1.000000000001], tolerance=1e-12)
    ValueError: All values of 'values' are approximately equal (≈1.0)
    """
    # Convert to numpy array
    try:
        arr = asarray(user_input)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name!r} to be array-like; "
            f"got {type(user_input).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e
    
    # Handle empty or single-element arrays
    if arr.size < 2:
        return user_input if not isinstance(user_input, ndarray) else arr
    
    # Check if constant
    if arr.ndim == 0:
        # Scalar - always "constant"
        return user_input if not isinstance(user_input, ndarray) else arr
    
    # Check using tolerance for floating point
    if issubdtype(arr.dtype, floating):
        first_val = arr[0]
        # For floats, use tolerance
        if allclose(arr, first_val, rtol=tolerance, atol=tolerance):
            raise ValueError(
                f"All values of {param_name!r} are approximately equal "
                f"(≈{first_val:g} within tolerance {tolerance})"
            )
    else:
        # For integers/other types, exact equality
        if all(arr == arr[0]):
            constant_value = arr[0]
            raise ValueError(
                f"All values of {param_name!r} are the same "
                f"({constant_value!r})"
            )
    
    # Return in appropriate format
    if isinstance(user_input, ndarray):
        return arr
    elif isinstance(user_input, list):
        return arr.tolist()
    else:
        # Preserve original type if possible
        return user_input if hasattr(user_input, '__iter__') else arr.tolist()