from numpy import isclose


def verify_lower_lte_upper(
    lower: int | float,
    upper: int | float,
    param_name_lower: str,
    param_name_upper: str
) -> tuple[int | float, int | float]:

    if not isinstance(lower, (int, float)):
        raise TypeError(
            f"Expected {param_name_lower!r} to be an integer or float; "
            f"got {type(lower).__name__}"
        )
    
    if not isinstance(upper, (int, float)):
        raise TypeError(
            f"Expected {param_name_upper!r} to be an integer or float; "
            f"got {type(upper).__name__}"
        )
    
    if lower >= upper:
        raise ValueError(
            f"Expected {param_name_lower!r} to be less than "
            f"{param_name_upper!r}; got {lower!r} and {upper!r}"
        )
        
    return lower, upper


def verify_step_size(
    start: float | int,
    stop: float | int,
    step: float | int,
    min_elements: int = 2,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12
) -> tuple[float, float, float]:
    """
    Validate step size with tolerance for floating point precision.
    
    Parameters
    ----------
    min_elements : int
        Minimum number of elements in the sequence.
    rel_tol, abs_tol : float
        Tolerances for floating point comparisons.
    """
    # Validate inputs
    for name, val in [("start", start), ("stop", stop), ("step", step)]:
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"Expected {name!r} to be numeric; got {type(val).__name__}"
            )
    
    # Step cannot be effectively zero
    if isclose(step, 0, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(f"step is effectively zero: {step}")
    
    # Calculate range and direction
    range_size = stop - start
    
    # Check step direction matches range
    if step * range_size < 0:  # Opposite signs
        raise ValueError(
            f"Step {step} moves away from stop {stop} instead of toward it "
            f"(start={start})"
        )
    
    # Calculate number of steps
    num_steps_float = range_size / step
    
    # Must have at least min_elements - 1 steps
    min_steps = min_elements - 1
    if num_steps_float < min_steps - rel_tol:
        raise ValueError(
            f"Step {step} too large for range [{start}, {stop}]. "
            f"Would produce less than {min_elements} elements."
        )
    
    return start, stop, step