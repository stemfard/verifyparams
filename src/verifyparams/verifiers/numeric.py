from typing import Any, Literal

from verifyparams.verifiers.dtypes import verify_int, verify_int_or_float
from verifyparams.verifiers.intervals import verify_lower_lte_upper


def verify_decimals(
    value: Any,
    a: int = -1,
    b: int = 14,
    param_name: str = "decimals: Decimal points"
) -> int:
    """
    Validate a `value` parameter (for rounding/formatting operations).
    
    Parameters
    ----------
    value : Any
        The `value` parameter to validate.
    a : int, optional (default=-1)
        Minimum allowed value for `value`.
    b : int, optional (default=14)
        Maximum allowed value for `value`.
    param_name : str, optional (default="decimals")
        Name of parameter for error messages.
    
    Returns
    -------
    int
        The validated value parameter.
    
    Raises
    ------
    TypeError
        If `value` is not an integer.
    NumericError
        If `value` is outside the allowed range.
    """
    verify_int(value=a, param_name="a")
    verify_int(value=b, param_name="b")
    
    verify_lower_lte_upper(
        lower=a, upper=b,
        param_name_lower="a",
        param_name_upper="b"
    )
    
    verify_int_or_float(value=value, param_name="value")
    
    verify_numeric(
        value=value,
        limits=[a, b],
        is_integer=True,
        param_name=param_name
    )
    
    return value


def verify_numeric(
    value: Any,
    limits: list[int | float] | None = None,
    boundary: Literal["inclusive", "exclusive"] = "inclusive",
    is_positive: bool = False,
    is_integer: bool = False,
    allow_none: bool = False,
    param_name: str = "value"
) -> int | float | None:
    """
    Validate and convert numeric input values.
    
    Parameters
    ----------
    value : Any
        The input value to validate.
    limits : list[int | float], optional
        List of [lower, upper] limits for the value.
    boundary : {'inclusive', 'exclusive'}, optional (default='inclusive')
        Whether limits include ('inclusive') or exclude ('exclusive') 
        boundaries.
    is_positive : bool, optional (default=False)
        If True, value must be positive (> 0).
    is_integer : bool, optional (default=False)
        If True, value must be (or be convertible to) an integer.
    allow_none : bool, optional (default=False)
        If True, None values are allowed and returned as None.
    param_name : str, optional (default='value')
        Name of the parameter for error messages.
    
    Returns
    -------
    int or float or None
        The validated and converted numeric value.
    
    Raises
    ------
    TypeError
        If input is not a number or cannot be converted.
    ValueError
        If input violates constraints (limits, positivity, etc.).
    """
    # Handle None values
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{param_name!r} cannot be None")
    
    # Try to convert to float first
    try:
        if isinstance(value, (int, float)):
            value = float(value)
        elif isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError(f"{value!r} cannot be empty")
            value = float(cleaned)
        else:
            value = float(value) # will fail if not convertable
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name!r} to be a number (integer or float), "
            f"got {type(value).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(
            f"Expected {param_name!r} to be a number (integer or float), "
            f"got {value!r}"
        ) from e
    
    # Check for positive requirement
    if is_positive and value <= 0:
        raise ValueError(
            f"Expected {param_name!r} to be positive (greater than 0), "
            f"got {value!r}"
        )
    
    # Check integer requirement
    if is_integer:
        if not value.is_integer():
            raise ValueError(
                f"Expected {param_name!r} to be an integer, got {value!r}"
            )
        value = int(value)
    
    # Check limits if provided
    if limits is not None:
        if len(limits) != 2:
            raise ValueError(
                "Expected 'limits' to be a list of two elements "
                f"[lower, upper], got {limits!r}"
            )
        
        verify_int_or_float(value=limits[0], param_name="limits[0]")
        verify_int_or_float(value=limits[1], param_name="limits[1]")
        
        lower, upper = min(limits), max(limits)
        
        if lower == upper:
            raise ValueError("'limits' must have different numeric values.")
        
        if boundary == 'inclusive':
            in_range = lower <= value <= upper
        elif boundary == 'exclusive':
            in_range = lower < value < upper
        else:
            raise ValueError(
                "Expected 'boundary' to be either 'inclusive' or "
                f"'exclusive', got {boundary!r}"
            )
        
        if not in_range:
            # Format error message based on boundary type
            if boundary == 'inclusive':
                error_msg = f"between {lower} and {upper} (inclusive)"
            else:
                error_msg = f"strictly between {lower} and {upper} (exclusive)"
            
            raise ValueError(
                f"Expected {param_name!r} to be {error_msg}, got {value!r}"
            )
    
    # Return appropriate type
    if is_integer:
        return int(value)
    elif value.is_integer():
        return int(value)
    else:
        return value