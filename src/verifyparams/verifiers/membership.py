from typing import Any, Literal, TypeVar, Generic

T = TypeVar('T', str, int, float)

class MembershipValidator(Generic[T]):
    """Helper for membership validation with caching."""
    
    def __init__(self, valid_items: list[T], case_sensitive: bool = True):
        self.valid_items = valid_items
        self.case_sensitive = case_sensitive
        
        # Create lookup dictionaries
        self._lookup = {item: item for item in valid_items}
        
        if not case_sensitive:
            # Add lowercase versions for strings
            for item in valid_items:
                if isinstance(item, str):
                    self._lookup[item.lower()] = item
    
    def validate(
        self, 
        user_input: Any,
        default: T | None = None, 
        param_name: str = "value"
    ) -> T:
        """Validate input against valid items."""
        # Try direct lookup
        if user_input in self._lookup:
            return self._lookup[user_input]
        
        # Try case-insensitive for strings
        if isinstance(user_input, str) and not self.case_sensitive:
            lookup_key = user_input.lower()
            if lookup_key in self._lookup:
                return self._lookup[lookup_key]
        
        # Try type conversion
        try:
            # Sample type from first valid item
            sample_type = type(self.valid_items[0])
            converted = sample_type(user_input)
            if converted in self._lookup:
                return self._lookup[converted]
        except (ValueError, TypeError):
            pass
        
        # Use default or raise error
        if default is not None:
            if default in self.valid_items:
                return default
            else:
                raise ValueError(f"Default {default!r} not in `valid_items`")
        
        # prepare error message
        k = 10
        items_str = ", ".join(repr(item) for item in self.valid_items[:k])
        if len(self.valid_items) > k:
            items_str += f", ..., {len(self.valid_items) - k} more"
        
        raise ValueError(
            f"Expected {param_name!r} to be one of: {items_str}, "
            f"got {user_input!r}"
        )


def verify_membership(
    user_input: Any,
    valid_items: list[str | int | float],
    case_sensitive: bool = True,
    default: str | int | float | None = None,
    param_name: str = "value"
) -> str | int | float:
    """
    Fast membership validation using cached validator.
    """
    # to be cached in future versions
    validator = MembershipValidator(
        valid_items=valid_items,
        case_sensitive=case_sensitive
    )
    
    return validator.validate(
        user_input=user_input,
        param_name=param_name,
        default=default
    )
    
    
def verify_membership_allowed(
    values: Any,
    allowed_values: list[Any],
    on_invalid: Literal["raise", "remove", "ignore"] = "raise",
    param_name: str = "values"
) -> list[Any]:
    """
    Filter input to keep only values that are in `allowed_values` with 
    configurable behavior.
    
    Parameters
    ----------
    values : Any
        Single value or list of values.
    allowed_values : List[Any]
        Values that are allowed.
    on_invalid : str
        What to do with invalid values:
        - "raise": raise ValueError
        - "remove": filter out invalid values
        - "ignore": keep invalid values (just validate)
    param_name : str
        Parameter name for errors.
    
    Returns
    -------
    list[any]
        Processed values.
    """
    # Convert to list
    if isinstance(values, str):
        input_list = [values]
    elif not hasattr(values, '__iter__'):
        input_list = [values]
    else:
        input_list = list(values)
    
    # Check membership
    allowed_set = set(allowed_values)
    invalid = [v for v in input_list if v not in allowed_set]
    
    if invalid:
        if on_invalid == "raise":
            if len(invalid) == 1:
                error_msg = f"The value {invalid[0]!r} is invalid"
            else:
                error_msg = (
                    f"The values {', '.join(repr(v) for v in invalid)} are "
                    "invalid"
                )
            
            k = 10
            allowed_str = ', '.join(repr(v) for v in allowed_values[:k])
            if len(allowed_values) > k:
                allowed_str += f", ..., {len(allowed_values) - k} more"
            
            raise ValueError(
                f"{param_name!r} {error_msg}. "
                f"Allowed values: {allowed_str}"
            )
        elif on_invalid == "remove":
            input_list = [v for v in input_list if v in allowed_set]
        else:
            pass # "ignore" does nothing
    
    return input_list