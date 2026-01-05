from typing import Any


def verify_boolean(
    value: Any,
    default: bool = False,
    accept_strings: bool = True,
    accept_numbers: bool = True
) -> bool:
    """
    Practical boolean validation with common conversions.
    
    Parameters
    ----------
    value : Any
        Input to interpret as boolean.
    default : bool, optional (default=False)
        Default value if input cannot be interpreted.
    accept_strings : bool, optional (default=True)
        Whether to accept string representations.
    accept_numbers : bool, optional (default=True)
        Whether to accept numeric representations.
    
    Returns
    -------
    bool
        Interpreted boolean value.
    """
    # Handle actual booleans
    if isinstance(value, bool):
        return value
    
    # Handle None
    if value is None:
        return default
    
    # Handle strings if allowed
    if accept_strings and isinstance(value, str):
        value_lower = value.lower().strip()
        true_values = {'true', 'yes', 'y', '1', 'on', 't', 'ok'}
        false_values = {'false', 'no', 'n', '0', 'off', 'f'}
        
        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False
        # If string doesn't match known values, continue to default
    
    # Handle numbers if allowed
    if accept_numbers and isinstance(value, (int, float)):
        if value == 0:
            return False
        elif value == 1:
            return True
        # Other numbers are ambiguous
    
    # Return default for uninterpretable input
    return default


def verify_int(value: Any, param_name: str) -> int:
    
    if not isinstance(value, int):
        raise TypeError(
            f"Expected {param_name!r} to be an integer; "
            f"got {type(value).__name__}"
        )
        
    return value
    

def verify_float(value: Any, param_name: str) -> float:
    
    if not isinstance(value, float):
        raise TypeError(
            f"Expected {param_name!r} to be an integer; "
            f"got {type(value).__name__}"
        )
        
    return value


def verify_int_or_float(value: Any, param_name: str) -> float:
    
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Expected {param_name!r} to be an integer or float; "
            f"got {type(value).__name__}"
        )
        
    return value


def verify_complex(value: Any, param_name: str) -> complex:
    
    if not isinstance(value, complex):
        raise TypeError(
            f"Expected {param_name!r} to be a complex number (a + bi); "
            f"got {type(value).__name__}"
        )
        
    return value


def verify_positive(
    value: Any, strict: bool = False, param_name: str = "value"
) -> int | float:
    """
    Verify that a value is numeric and positive.

    Parameters
    ----------
    value : Any
        The value to check.
    strict : bool, optional (default=False)
        If True, zero is considered invalid (strictly positive).
    param_name : str, optional
        Name of the parameter (used in error messages).

    Returns
    -------
    int | float
        The verified numeric value.

    Raises
    ------
    TypeError
        If `value` is not numeric.
    ValueError
        If `value` is not positive (or zero if strict=True).
    """
    # Verify numeric type
    verify_int_or_float(value=value, param_name=param_name)

    # Check positivity
    msg = f"Expected {param_name!r} to be positive; got {value}"
    if strict:
        if value <= 0:
            raise ValueError(msg.replace("positive", "strictly positive"))
    else:
        if value < 0:
            raise ValueError(msg)

    return value