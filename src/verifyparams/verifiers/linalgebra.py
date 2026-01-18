from typing import Any

from numpy import asarray, ndarray
from pandas import DataFrame
from sympy import Matrix, MatrixBase, SympifyError, sympify


def verify_array_or_matrix(
    A: Any,
    nrows: int | None = None,
    ncols: int | None = None,
    is_square: bool = False,
    to_matrix: bool = False,
    param_name: str = 'array'
) -> ndarray | Matrix:
    """
    Validate array/matrix with type preservation options.
    
    Parameters
    ----------
    A : Any
        Input array/matrix.
    nrows, ncols : int, optional
        Expected dimensions.
    is_square : bool
        Whether to require square matrix.
    to_matrix : bool
        If True and input is convertible, returns SymPy Matrix.
    param_name : str
        Parameter name for errors.
    
    Returns
    -------
    ndarray or Matrix
        Validated array/matrix.
    """
    original_type = type(A)
    
    if isinstance(A, (ndarray, MatrixBase)):
        result = A
    elif isinstance(A, DataFrame):
        result = A.values
    else:
        try:
            if to_matrix:
                try:
                    result = Matrix(A)
                except (TypeError, ValueError, AttributeError, SympifyError):
                    result = Matrix(sympify(A))
            else:
                result = asarray(A)
        except (TypeError, ValueError, AttributeError, SympifyError) as e:
            raise TypeError(
                f"Expected {param_name!r} to be array-like, Matrix, "
                f"or DataFrame, got {original_type.__name__}"
            ) from e
    
    shape = result.shape
    
    if len(shape) != 2:
        raise ValueError(
            f"Expected {param_name!r} to be 2-dimensional, "
            f"got shape = {shape}"
        )
    
    if nrows is not None and shape[0] != nrows:
        raise ValueError(
            f"Expected {param_name!r} to have {nrows} rows, got {shape[0]}"
        )
    
    if ncols is not None and shape[1] != ncols:
        raise ValueError(
            f"Expected {param_name!r} to have {ncols} columns, got {shape[1]}"
        )
    
    if is_square and shape[0] != shape[1]:
        raise ValueError(
            f"Expected {param_name!r} to be square, got shape = {shape}"
        )
        
    if to_matrix:
        return result if isinstance(result, MatrixBase) else Matrix(result)

    return result


def verify_square(
    A: ndarray | Matrix,
    tolerance: float = 1e-12,
    param_name: str = "A"
) -> ndarray | Matrix:
    """
    Fast square matrix validation for numpy arrays.
    
    Parameters
    ----------
    A : ndarray
        Input array.
    param_name : str
        Parameter name.
    tolerance : float
        Tolerance for near-square matrices.
    
    Returns
    -------
    ndarray
        Validated square matrix.
    """
    is_sympy_matrix = True if isinstance(A, MatrixBase) else False
    
    if not isinstance(A, ndarray):
        try:
            A = asarray(A)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Expected {param_name!r} to be array-like, "
                f"got {type(A).__name__}"
            ) from e
    
    if A.ndim != 2:
        raise ValueError(
            f"Expected {param_name!r} to be a 2D matrix / array, "
            f"got {A.ndim}D array"
        )
    
    rows, cols = A.shape
    if abs(rows - cols) > tolerance:
        raise ValueError(
            f"Expected {param_name!r} to be square, got shape ({rows}, {cols})"
        )
    
    return Matrix(A) if is_sympy_matrix else A


def verify_linear_system(
    A: Any,
    b: Any,
    require_square: bool = False,
    param_name_A: str = "A",
    param_name_b: str = "b"
) -> tuple[ndarray, ndarray]:
    """
    Validate linear system Ax = b.
    
    Parameters
    ----------
    A : array_like
        Coefficient matrix.
    b : array_like
        Constant vector.
    require_square : bool
        If True, requires A to be square.
    param_name_A, param_name_b : str
        Parameter names for error messages.
    
    Returns
    -------
    tuple[ndarray, ndarray]
        Validated (A, b).
    """
    # Convert to arrays
    is_sympy_matrix = True if isinstance(A, MatrixBase) else False
    
    try:
        A = asarray(A)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name_A!r} to be a numeric array, "
            f"got {type(A).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e
        
    try:
        b = asarray(b)
    except TypeError as e:
        raise TypeError(
            f"Expected {param_name_b!r} to be a numeric array, "
            f"got {type(b).__name__}"
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e
    
    # Validate dimensions
    if A.ndim != 2:
        raise ValueError(
            f"Expected {param_name_A!r} to be a 2D matrix, got {A.ndim}D array"
        )
    
    # Handle b as vector (allow column or row vector)
    if b.ndim == 2:
        if b.shape[1] == 1:
            b = b.flatten()  # Column vector to 1D
        elif b.shape[0] == 1:
            b = b.flatten()  # Row vector to 1D
        else:
            raise ValueError(
                f"Expected {param_name_b!r} to be a vector (1D array), "
                f"got 2D array with shape {b.shape}"
            )
    elif b.ndim != 1:
        raise ValueError(
            f"Expected {param_name_b!r} to be a vector (1D array), "
            f"got {b.ndim}D array"
        )
    
    # Check compatibility
    a_rows, a_cols = A.shape
    b_len = len(b)
    
    if a_rows != b_len:
        raise ValueError(
            f"Expected number of rows of {param_name_A!r} to be equal to the "
            f"number of elements of {param_name_b!r}, got "
            f"{param_name_A!r} ({A.shape}) vs {param_name_b!r} ({b.shape})"
        )
    
    # Optional square matrix check
    if require_square and a_rows != a_cols:
        raise ValueError(
            f"Expected {param_name_A!r} to be a square array/matrix, "
            f"got shape ({a_rows}, {a_cols})"
        )
        
    if is_sympy_matrix:
        A, b = Matrix(A), Matrix(b)
    
    return A, b