import copy
from dataclasses import field
from typing import Any, Iterator

class Result:
    """
    Generic result container for any function.

    - Attributes are created dynamically based on what the function returns.
    - Supports dot access: res.attr
    - Provides .keys(), .items(), .values(), and a summary method.
    """
    
    _data: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __init__(self, **kwargs: Any):
        # Initialize the _data field
        object.__setattr__(self, '_data', {})
        
        # Add all items as attributes (stored in _data dict)
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def to_dict(self) -> dict[str, Any]:
        """Return all values as dictionary"""
        # Return a copy to prevent modification
        return copy.deepcopy(self._data)

    def keys(self) -> Iterator[str]:
        """Return all attribute names."""
        return iter(self._data.keys())

    def values(self) -> Iterator[Any]:
        """Return all attribute values."""
        return iter(self._data.values())

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return all attribute-name / value pairs."""
        return iter(self._data.items())

    def summary(self) -> None:
        """Pretty-print all stored attributes."""
        for key, value in self.items():
            print(f"{key}: {value}")

    def __repr__(self) -> str:
        """Print the attributes"""
        attrs = ", ".join(self._data.keys())
        return f"Result({attrs})"

    # Optional: Add helpful methods that dataclasses provide
    def __len__(self) -> int:
        """Return number of attributes."""
        return len(self._data)