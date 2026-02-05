from numpy import ndarray


def str_data_join(
    data: list | tuple | ndarray,
    delim: str = ", ",
    is_quoted: bool = False,
    use_map: bool = True,
    use_and: bool = False
) -> str:
    """
    Join a sequence of values into a string.
    
    Parameters
    ----------
    values : list, tuple, or ndarray
        Values to join.
    delim : str, default ", "
        Delimiter between values.
    is_quoted : bool, default False
        Whether to quote each value.
    use_map : bool, default True
        Whether to use map() for faster joining.
    use_and : bool, default False
        Whether to use "and" before the last item.
    
    Returns
    -------
    str
        Joined string.
    """
    if len(data) == 0:
        return ""
    
    if len(data) == 1:
        val = str(data[0])
        return f'"{val}"' if is_quoted else val
    
    if isinstance(data, ndarray):
        data = data.tolist() 
    
    if use_and and len(data) > 1:
        data_copy = list(data)
        if len(data_copy) > 2:
            data_copy.insert(-1, "and")
            delim = ", "
        else:
            return f"{data_copy[0]} and {data_copy[1]}"
        data = data_copy
    
    if use_map:
        if is_quoted:
            data_str = map(lambda x: f"'{x}'", data)
        else:
            data_str = map(str, data)
        strng = delim.join(data_str)
    else:
        if is_quoted:
            data_str = [f"'{v}'" for v in data]
        else:
            data_str = [str(v) for v in data]
        strng = delim.join(data_str)
        
    strng = strng.replace(", 'and',", " and").replace(", and,", " and")
    
    return strng

    
def str_data_join_contd(
    data: list | tuple | ndarray,
    max_show: int = 10,
    use_map: bool = True,
    is_quoted: bool = False
) -> str:
    """
    Join a sequence of values into a string with truncation.
    
    Parameters
    ----------
    data : list, tuple, or ndarray
        Values to join.
    max_show : int
        Maximum number of values to show before truncating
    use_map : bool, default True
        Whether to use map() for faster joining.
    is_quoted : bool, default False
        Whether to quote each value.
    
    Returns
    -------
    data_str : str
        Formatted string of values.
    """
    if len(data) == 0:
        return ""
    
    kwargs = {
        "use_map": use_map,
        "is_quoted": is_quoted
    }
    
    if len(data) <= max_show or len(data) <= 10:
        data_str = str_data_join(data, **kwargs)
    else:
        first_data = str_data_join(data[:5], **kwargs)
        last_data = str_data_join(data[-3:], **kwargs)
        
        data_str = f"{first_data}, ..., {last_data}"
        
    return data_str