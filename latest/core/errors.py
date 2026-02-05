SYMPIFY_ERRORS = (
    SyntaxError,
    NameError,
    TypeError,
    ValueError,
    AttributeError,
    ImportError
)


class NonSquareArray(ValueError):
    pass

class DimensionsError(ValueError):
    pass

class CompatibilityError(ValueError):
    pass