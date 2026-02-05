from typing import Any

from numpy import asarray, ndarray, float64
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sympy import Matrix, MatrixBase, SympifyError, sympify

from verifyparams.core.errors import (
    NonSquareArray, DimensionsError, CompatibilityError
)


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
    elif isinstance(A, (Series, DataFrame)):
        result = A.to_numpy()
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
                f"or DataFrame, got {original_type.__name__}: {str(e)}"
            ) from e
    
    shape = result.shape
    
    if len(shape) != 2:
        raise ValueError(
            f"Expected {param_name!r} to be 2-dimensional, "
            f"got shape = {shape}"
        )
    
    if is_square and shape[0] != shape[1]:
        raise NonSquareArray(
            f"Expected {param_name!r} to be square, got shape = {shape}"
        )
        
    if nrows is not None and shape[0] != nrows:
        raise ValueError(
            f"Expected {param_name!r} to have {nrows} rows, got {shape[0]}"
        )
    
    if ncols is not None and shape[1] != ncols:
        raise ValueError(
            f"Expected {param_name!r} to have {ncols} columns, got {shape[1]}"
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
    param_names: tuple[str, str] = ("A", "b")
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
        param_name_A, param_name_b = param_names
    except (ValueError, TypeError):
        param_name_A, param_name_b = ("A", "b")
        
    def validate_arr(A: Any, param_name: str) -> NDArray[float64]:
        
        try:
            arr = asarray(A, dtype=float)
        except TypeError as e:
            raise TypeError(
                f"Expected {param_name!r} to be a numeric array, "
                f"got {type(A).__name__}"
            ) from e
        except ValueError as e:
            if "could not convert string to float" in str(e):
                raise ValueError(
                    f"Expected all values of {param_name!r} to be numeric, "
                    "got at least one non-numeric value"
                )
            raise ValueError(str(e)) from e
        
        return arr
    
    A = validate_arr(A, param_name=param_name_A)
    b = validate_arr(b, param_name=param_name_b)
    
    # Validate dimensions
    if A.ndim != 2:
        raise DimensionsError(
            f"Expected {param_name_A!r} to be a 2D matrix, got {A.ndim}D array"
        )
    
    try:
        b = b.flatten().reshape(-1, 1) # convert to column vector
    except (TypeError, ValueError, AttributeError) as e:
        raise ValueError(
            f"Expected {param_name_b!r} to be a vector (1D array), "
            f"got {b.ndim}D array"
        ) from e
    
    nrows_a, ncols_a = A.shape
    len_b = len(b)
    
    # check square
    if require_square and nrows_a != ncols_a:
        raise NonSquareArray(
            f"Expected {param_name_A!r} to be a square array/matrix, "
            f"got shape ({nrows_a}, {ncols_a})"
        )
        
    # Check compatibility    
    if nrows_a != len_b:
        raise CompatibilityError(
            f"Expected number of rows of {param_name_A!r} to be equal to the "
            f"number of elements of {param_name_b!r}, got "
            f"{param_name_A!r} {A.shape} vs {param_name_b!r} {b.shape}"
        )
        
    if is_sympy_matrix:
        A, b = Matrix(A), Matrix(b)
    
    return A, b