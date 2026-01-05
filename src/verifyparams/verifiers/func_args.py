def verify_args_count(
    ftn_name: str, args_dict: dict, required_count: int = None    
) -> None:
    """
    Validate that exactly `required_count` out of all arguments are 
    provided. If `required_count` is not specified, validates that all 
    arguments are provided.

    Parameters
    ----------
    ftn_name : str
        The name of the function being validated.
    args_dict : dict
        A dictionary containing the function arguments (keys) and their 
        respective values (values).
    required_count : int, optional
        The exact number of arguments that should be provided. If None,
        all arguments must be provided.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the number of provided parameters doesn't match 
        `required_count`.

    Notes
    -----
    This function is typically used to ensure that exactly 
    `required_count` out of all possible parameters are supplied, 
    which is essential for performing calculations that require `n` 
    known values to determine the other.
    """
    args_names = list(args_dict.keys())
    total_args = len(args_names)
    provided_count = sum(value is not None for value in args_dict.values())
    
    # If required_count is not specified, default to all arguments
    if required_count is None:
        required_count = total_args
    
    if provided_count != required_count:
        # Create a list of provided argument names for better error messages
        provided_args = [
            name for name, value in args_dict.items() if value is not None
        ]
        missing_args = [
            name for name, value in args_dict.items() if value is None
        ]
        
        raise ValueError(
            f"Expected exactly {required_count} out of {total_args} arguments "
            f"to be provided for '{ftn_name}()'; got {provided_count}.\n"
            f"Provided: {provided_args}\n"
            f"Missing: {missing_args}"
        )