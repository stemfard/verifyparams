import re
from typing import Literal, Pattern


def verify_string(
    value: str,
    allow_empty: bool = False,
    min_length: int = 0,
    max_length: int | None = None,
    allowed_chars: str | None = None,
    pattern: Pattern | None = None,
    pattern_str: str | None = None,
    str_case: Literal["lower", "upper", "title"] | None = None,
    strip_whitespace: bool = False,
    param_name: str = "value"
) -> str:
    """
    Comprehensive string validation with constraints and pattern matching.
    
    Parameters
    ----------
    value : str
        String to validate.
    allow_empty : bool, optional
        Allow empty strings.
    min_length : int, optional
        Minimum string length.
    max_length : int, optional
        Maximum string length.
    allowed_chars : str, optional
        String containing allowed characters.
    pattern : Pattern, optional
        Compiled regex pattern to match.
    pattern_str : str, optional
        Regex pattern string (compiled if pattern not provided).
    case : {'lower', 'upper', 'title', None}, optional
        Case normalization to apply.
    strip_whitespace : bool, optional
        Strip leading/trailing whitespace before validation.
    param_name : str, optional
        Parameter name for error messages.
    
    Returns
    -------
    str
        Validated string.
    
    Raises
    ------
    TypeError
        If input is not a string.
    ValueError
        If string violates constraints.
    
    Examples
    --------
    >>> verify_string("abc123", pattern_str=r'^[a-z0-9]+$')
    'abc123'
    
    >>> verify_string("  test  ", strip_whitespace=True)
    'test'
    
    >>> verify_string("UPPERCASE", case='lower')
    'uppercase'
    """
    # Type check
    if not isinstance(value, str):
        raise TypeError(
            f"Expected {param_name!r} to be a string, "
            f"got {type(value).__name__}"
        )
    
    # Apply preprocessing
    result = value
    
    if strip_whitespace:
        result = result.strip()
    
    # Check empty after stripping
    if not result and not allow_empty:
        raise ValueError(
            f"{param_name!r} cannot be empty when 'allow_empty' is False"
        )
    
    # Early return for empty string (if allowed)
    if not result:
        return result
    
    # Check length constraints
    length = len(result)
    
    if length < min_length:
        s = "" if min_length == 1 else "s"
        raise ValueError(
            f"Expected {param_name!r} to have at least {min_length} "
            f"character{s}, got {length} character{s}"
        )
    
    if max_length is not None and length > max_length:
        raise ValueError(
            f"Expected {param_name!r} to have at most {max_length} "
            f"characters, got {length} characters"
        )
    
    # Check allowed characters
    if allowed_chars is not None:
        invalid_chars = []
        for char in result:
            if char not in allowed_chars:
                invalid_chars.append(char)
        
        if invalid_chars:
            # Show unique invalid characters
            unique_invalid = set(invalid_chars)
            k = 10
            if len(unique_invalid) <= k:
                invalid_str = ", ".join(repr(c) for c in unique_invalid)
                error_msg = f"invalid characters: {invalid_str}"
            else:
                invalid_str = (
                    ", ".join(repr(c) for c in list(unique_invalid)[:k])
                )
                error_msg = f"invalid characters including: {invalid_str}"
            
            raise ValueError(f"{param_name!r} contains {error_msg}")
    
    # Check pattern
    if pattern is not None or pattern_str is not None:
        # Compile pattern if string provided
        if pattern is None:
            try:
                pattern = re.compile(pattern_str)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        
        if not pattern.fullmatch(result):
            raise ValueError(
                f"Expected {param_name!r} to match pattern {pattern.pattern!r}"
            )
    
    # Apply case normalization
    if str_case == 'lower':
        result = result.lower()
    elif str_case == 'upper':
        result = result.upper()
    elif str_case == 'title':
        result = result.title()
    elif str_case is not None and str_case not in ["lower", "upper", "title"]:
        raise ValueError(
            f"Expected {str_case!r} to be one of: 'lower', 'upper' or "
            f"'title', got {str_case!r}"
        )
    
    return result


def verify_str_email(
    value: str,
    param_name: str = "email",
) -> str:
    """Validate email address."""
    return verify_string(
        value,
        pattern_str=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        case='lower',
        strip_whitespace=True,
        allow_empty=False,
        param_name=param_name
    )


def verify_str_identifier(
    value: str,
    param_name: str = "identifier"
) -> str:
    """Validate Python identifier."""
    return verify_string(
        value,
        pattern_str=r'^[a-zA-Z_][a-zA-Z0-9_]*$',
        strip_whitespace=True,
        allow_empty=False,
        param_name=param_name
    )
    

def verify_str_alphanumeric(
    value: str,
    allow_empty: bool = False,
    param_name: str = "value"
) -> str:
    """Validate alphanumeric string."""
    return verify_string(
        value,
        pattern_str=r'^[a-zA-Z0-9]+$',
        allow_empty=allow_empty,
        param_name=param_name
    )


def verify_str_numeric(
    value: str,
    allow_empty: bool = False,
    param_name: str = "value"
) -> str:
    """Validate numeric string (digits only)."""
    return verify_string(
        value,
        pattern_str=r'^[0-9]+$',
        allow_empty=allow_empty,
        param_name=param_name
    )