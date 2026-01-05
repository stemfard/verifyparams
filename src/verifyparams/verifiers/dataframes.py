from typing import Any

from pandas import DataFrame, Series


def verify_axis(
    value: Any,
    param_name: str = 'axis',
    allow_integers: bool = True,
    allow_strings: bool = True,
    allow_booleans: bool = True,
    allow_none: bool = True
) -> int | None:
    """
    Validate array axis parameter with flexible input options.
    
    Parameters
    ----------
    value : Any
        Axis specification to validate.
    param_name : str, optional (default='axis')
        Name of parameter for error messages.
    allow_integers : bool, optional (default=True)
        Whether to accept integer inputs (0, 1).
    allow_strings : bool, optional (default=True)
        Whether to accept string inputs ('rows', 'columns', etc.).
    allow_booleans : bool, optional (default=True)
        Whether to accept boolean inputs (True=1, False=0).
    allow_none : bool, optional (default=True)
        Whether to accept None as a valid input.
    
    Returns
    -------
    int or None
        Validated axis (0 for rows/index, 1 for columns), or None.
    
    Raises
    ------
    ValueError
        If value is not a valid axis specification.
    
    Examples
    --------
    >>> verify_axis(0)
    0
    >>> verify_axis('rows')
    0
    >>> verify_axis(True)
    1
    >>> verify_axis(None)
    None
    >>> verify_axis(2)
    ValueError: Expected 'axis' to be 0 or 1; got 2
    """
    # Handle None
    if value is None:
        if allow_none:
            return None
        else:
            raise ValueError(
                f"{param_name!r} cannot be `None`. To allow `None` values, "
                f"set `allow_none` to `True`"
            )
    
    # Handle booleans
    if allow_booleans and isinstance(value, bool):
        return 1 if value else 0
    
    # Handle integers
    if allow_integers and isinstance(value, int):
        if value == 0:
            return 0
        elif value == 1:
            return 1
        else:
            raise ValueError(
                f"Expected {param_name!r} to be 0 or 1; got {value}"
            )
    
    # Handle strings
    if allow_strings and isinstance(value, str):
        axis_str = value.lower().strip()
        
        # Row/index variants
        row_keywords = {'0', 'row', 'rows', 'index', 'indices'}
        if allow_booleans:
            row_keywords.add('false')
        
        # Column variants  
        col_keywords = {'1', 'col', 'column', 'columns'}
        if allow_booleans:
            col_keywords.add('true')
        
        if axis_str in row_keywords:
            return 0
        elif axis_str in col_keywords:
            return 1
        else:
            # Try numeric conversion for strings like "0.0", "1.0"
            try:
                # Try float first to catch "0.0", then convert to int
                num_val = float(axis_str)
                if num_val.is_integer():
                    int_val = int(num_val)
                    if int_val == 0:
                        return 0
                    elif int_val == 1:
                        return 1
            except ValueError:
                pass
    
    # Invalid input - build helpful error message
    valid_options = []
    if allow_integers:
        valid_options.extend(['0', '1'])
    if allow_strings:
        valid_options.extend(["'rows'", "'columns'", "'index'"])
    if allow_booleans:
        valid_options.extend(['True', 'False'])
    if allow_none:
        valid_options.append('None')
    
    options_str = ', '.join(valid_options)
    
    raise ValueError(
        f"Expected {param_name!r} to be one of: {options_str}; "
        f"got {value!r}"
    )
    
    
def verify_dframe(
    user_input: Any,
    convert: bool = True,
    require_columns: list[int | str] | None = None,
    require_index: bool = False,
    param_name: str = 'dataframe'
) -> DataFrame:
    """
    Validate and convert input to pandas DataFrame.
    
    Parameters
    ----------
    user_input : Any
        Input to validate. Can be:
        - pandas DataFrame
        - pandas Series
        - 2D array-like (list of lists, numpy array)
        - Dict of {column: values}
        - List of dicts
    convert : bool, optional (default=True)
        If True, attempts to convert non-DataFrame inputs.
        If False, only accepts DataFrame inputs.
    require_columns : list, optional
        List of column names that must be present.
    require_index : bool, optional (default=False)
        If True, requires the DataFrame to have a meaningful index.
    param_name : str, optional (default='dataframe')
        Parameter name for error messages.
    
    Returns
    -------
    DataFrame
        Validated pandas DataFrame.
    
    Raises
    ------
    TypeError
        If input cannot be converted to DataFrame.
    ValueError
        If DataFrame doesn't meet requirements.
    
    Examples
    --------
    >>> verify_dframe({'a': [1, 2], 'b': [3, 4]})
    DataFrame with columns 'a' and 'b'
    
    >>> verify_dframe([1, 2, 3], convert=False)
    TypeError: 'dataframe' must be a DataFrame
    """
    # Already a DataFrame
    if isinstance(user_input, DataFrame):
        df = user_input
    
    # Try to convert if allowed
    elif convert:
        try:
            df = DataFrame(user_input)
        except TypeError as e:
            raise TypeError(
                f"Expected {param_name!r} to be a DataFrame; "
                f"got {type(user_input).__name__}"
            ) from e
        except ValueError as e:
            raise ValueError(
                f"{param_name!r} cannot be converted to a DataFrame: {e}"
            ) from e
    else:
        raise TypeError(
            f"Expected {param_name!r} to be a DataFrame; "
            f"got {type(user_input).__name__}"
        ) from e
    
    # Validate requirements
    if require_columns is not None:
        missing_cols = [col for col in require_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{param_name!r} missing required columns: {missing_cols}"
            )
    
    if require_index and df.index.isna().all():
        raise ValueError(
            f"Expected {param_name!r} to have a meaningful index; "
            f"got no index"
        )
    
    return df


def verify_series(
    user_input: Any,
    convert: bool = True,
    allow_scalar: bool = False,
    require_name: bool = False,
    dropna: bool = False,
    param_name: str = 'series'
) -> Series:
    """
    Validate and convert input to pandas Series.
    
    Parameters
    ----------
    user_input : Any
        Input to validate. Can be:
        - pandas Series
        - List, tuple, numpy array
        - Dict (keys become index)
        - Scalar (if allow_scalar=True)
    convert : bool, optional (default=True)
        If True, attempts to convert non-Series inputs.
        If False, only accepts Series inputs.
    allow_scalar : bool, optional (default=False)
        If True, allows single scalar values.
    require_name : bool, optional (default=False)
        If True, requires the Series to have a name.
    dropna : bool, optional (default=False)
        If True, drops NaN values from the Series.
    param_name : str, optional (default='series')
        Parameter name for error messages.
    
    Returns
    -------
    Series
        Validated pandas Series.
    
    Raises
    ------
    TypeError
        If input cannot be converted to Series.
    ValueError
        If Series doesn't meet requirements.
    
    Examples
    --------
    >>> verify_series([1, 2, 3])
    Series with values [1, 2, 3]
    
    >>> verify_series(5, allow_scalar=True)
    Series with single value 5
    """
    msg_type = (
        f"Expected {param_name!r} to be a Series; "
        f"got {type(user_input).__name__}"
    )
    msg_value = f"{param_name!r} cannot be converted to a Series: {e}"
    
    # Already a Series
    if isinstance(user_input, Series):
        s = user_input
    
    # Try to convert if allowed
    elif convert:
        # Handle scalar if allowed
        if allow_scalar and not isinstance(user_input, (list, tuple, dict)):
            try:
                s = Series([user_input])
            except TypeError as e:
                raise TypeError(msg_type) from e
            except ValueError as e:
                raise ValueError(msg_value) from e
        else:
            try:
                s = Series(user_input)
            except TypeError as e:
                raise TypeError(msg_type) from e
            except ValueError as e:
                raise ValueError(msg_value) from e
    else:
        raise TypeError(msg_type)
    
    # Apply transformations
    if dropna:
        s = s.dropna()
    
    # Validate requirements
    if require_name and s.name is None:
        raise ValueError(
            f"Expected {param_name!r} to have a name; got no name"
        )
    
    return s


def verify_df_columns(
    names: Any,
    allow_duplicates: bool = False,
    param_name: str = "columns"
) -> list[str]:
    """
    Validate DataFrame column names.
    
    Parameters
    ----------
    names : Any
        Column names to validate.
    allow_duplicates : bool
        Allow duplicate column names.
    param_name : str
        Parameter name.
    
    Returns
    -------
    list[str]
        Validated column names.
    """
    # Handle None
    if names is None:
        raise ValueError(
            f"Expected {param_name!r} to be a list of column names; got None"
        )
    
    # Handle single string
    if isinstance(names, str):
        return [names]
    
    # Handle list/tuple
    if isinstance(names, (list, tuple)):
        names_list = list(names)
    else:
        # Try to convert
        try:
            names_list = list(names)
        except TypeError:
            raise TypeError(
                f"Expected {param_name!r} to be a string or a list of "
                f"strings; got {type(names).__name__}"
            )
    
    # Validate all are strings
    for i, name in enumerate(names_list):
        if not isinstance(name, str):
            raise TypeError(
                f"Expected {param_name!r}[{i}] to be a string; "
                f"got {type(name).__name__}"
            )
        if not name:  # Empty string
            raise ValueError(f"{param_name!r}[{i}] cannot be empty string")
    
    # Check duplicates
    if not allow_duplicates and len(names_list) != len(set(names_list)):
        raise ValueError(
            "Expected 'param_name' to be unique; got duplicated values"
        )
    
    return names_list