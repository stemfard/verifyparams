from typing import Any

from math import isclose


def verify_sta_conf_level(
    level: Any,
    param_name: str = "conf_level"
) -> float:
    """
    Validate common confidence levels.
    
    Accepts: 0.90, 0.95, 0.99 or 90, 95, 99 (as percentages)
    """
    # Convert to float
    try:
        value = float(level)
    except (ValueError, TypeError):
        raise TypeError(
            f"Expected {param_name!r} to be numeric, "
            f"got {type(value).__name__}"
        )
    
    # Handle percentage
    if 1 <= value <= 100:
        value = value / 100.0
    
    # Check against allowed values
    allowed = {0.80, 0.85, 0.90, 0.95, 0.99, 0.995}
    
    for allowed_val in allowed:
        if isclose(a=value, b=allowed_val, rel_tol=1e-9):
            return allowed_val
    
    # Not allowed value(s) found
    allowed_str = ", ".join(str(v) for v in sorted(allowed))
    raise ValueError(
        f"Expected {param_name!r} to be one of: {allowed_str}, "
        f"got {value}"
    )


def verify_sta_sig_level(
    level: Any,
    param_name: str = "sig_level"
) -> float:
    """
    Validate common significance levels (alpha).
    
    Accepts: 0.10, 0.05, 0.01 or 10, 5, 1 (as percentages)
    """
    # Convert to float
    try:
        value = float(level)
    except (ValueError, TypeError):
        raise TypeError(
            f"Expected {param_name!r} to be numeric, "
            f"got {type(level).__name__}"
        )
    
    # Handle percentage
    if 1 <= value <= 100:
        value = value / 100.0
    
    # Check against allowed values
    allowed = {0.10, 0.05, 0.01}
    
    for allowed_val in allowed:
        if isclose(a=value, b=allowed_val, rel_tol=1e-9):
            return allowed_val
    
    # Not allowed value(s) found
    allowed_str = ", ".join(str(v) for v in sorted(allowed))
    raise ValueError(
        f"Expected {param_name!r} to be one of: {allowed_str}, "
        f"got {value}"
    )
    

def verify_sta_alternative(
    alternative: Any,
    param_name: str = "alternative"
) -> str:
    """
    Validate alternative hypothesis options.
    
    Accepts: 'less', 'two-sided', 'greater'
    """
    # Convert to string if not already
    if not isinstance(alternative, str):
        try:
            alternative = str(alternative)
        except TypeError as e:
            raise TypeError(
                f"Expected {param_name!r} to be a string, "
                f"got {type(alternative).__name__}"
            ) from e
        except ValueError as e:
            raise ValueError(str(e)) from e   
        
    
    # Normalize (lowercase, strip whitespace)
    alt = alternative.lower().strip()
    
    # Define valid alternatives
    allowed = {'less', 'two-sided', 'greater'}
    
    # Check exact match
    if alt in allowed:
        return alt
    
    # Check common aliases
    aliases = {
        'two.sided': 'two-sided',
        'two sided': 'two-sided',
        'two-tailed': 'two-sided',
        'two.tailed': 'two-sided',
        'two tailed': 'two-sided',
        'lower': 'less',
        'smaller': 'less',
        'left': 'less',
        'higher': 'greater',
        'larger': 'greater',
        'right': 'greater',
    }
    
    if alt in aliases:
        return aliases[alt]
    
    # Not a valid alternative
    allowed_str = ", ".join(f"{v}" for v in sorted(allowed))
    raise ValueError(
        f"Expected {param_name!r} to be one of: {allowed_str}, "
        f"got {alternative!r}"
    )