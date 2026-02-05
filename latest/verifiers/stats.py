from typing import Any


def verify_sta_conf_level(
    value: Any,
    param_name: str = "conf_level: Confidence level"
) -> float:
    """
    Validate common confidence levels.
    """
    try:
        value = float(value)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name!r} to be numeric, "
            f"got {type(value).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e
    
    # Handle percentage
    if 1 <= value <= 100:
        value = value / 100.0
    
    allowed = {0.80, 0.85, 0.90, 0.95, 0.99, 0.995}
    
    if value not in allowed:
        allowed_str = ", ".join(map(str, allowed))
        raise ValueError(
            f"Expected {param_name!r} to be one of: {allowed_str}, "
            f"got {value}"
        )
    
    return value


def verify_sta_sig_level(
    value: Any,
    param_name: str = "sig_level: Significance level"
) -> float:
    """
    Validate common significance levels (alpha).
    """
    try:
        value = float(value)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name!r} to be numeric, "
            f"got {type(value).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e
    
    # Handle percentage
    if 1 <= value <= 100:
        value = value / 100.0
    
    allowed = {0.10, 0.05, 0.01}
    
    if value not in allowed:
        allowed_str = ", ".join(map(str, allowed))
        raise ValueError(
            f"Expected {param_name!r} to be one of: {allowed_str}, "
            f"got {value}"
        )
        
    return value
    
    
def verify_sta_alternative(
    value: Any,
    param_name: str = "alternative: Alternative hypothesis"
) -> str:
    """
    Validate alternative hypothesis options.
    
    Accepts: 'less', 'two-sided', 'greater'
    """
    alternative = value
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
    
    alt = alternative.lower().strip() # Normalize (lowercase, strip whitespace)
    allowed = {'less', 'two-sided', 'greater'}
    
    if alt not in allowed:
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
        
        if alt not in aliases:
            allowed_str = ", ".join(allowed)
            raise ValueError(
                f"Expected {param_name!r} to be one of: {allowed_str}, "
                f"got {alternative!r}"
            )    
        
        return aliases[alt]
    
    return alt
    

def verify_sta_decision(
    value: Any,
    param_name: str = "decision: Decision method"
) -> float:
    """
    Validate decision methods.
    """
    allowed = ("test-stat", "p-value", "graph")
    if value not in allowed:   
        allowed_str = ", ".join(map(str, allowed))
        raise ValueError(
            f"Expected {param_name!r} to be one of: {allowed_str}, "
            f"got {value}"
        )
    
    return value


def verify_sta_decision(
    value: Any,
    param_name: str = "decision: Decision method"
) -> float:
    """
    Validate decision methods.
    """
    allowed = ("test-stat", "p-value", "graph")
    if value not in allowed:   
        allowed_str = ", ".join(map(str, allowed))
        raise ValueError(
            f"Expected {param_name!r} to be one of: {allowed_str}, "
            f"got {value}"
        )
    
    return value
    