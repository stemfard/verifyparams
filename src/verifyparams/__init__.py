"""
============
verifyparams
============

`verifyparams` is a Python library for validating and verifying function
parameters, arrays, matrices, dataframes, and other input types in
scientific, engineering and data-intensive applications."

See the webpage for more information and documentation:

    https://verifyparams.stemfard.org
"""

import sys

if sys.version_info < (3, 9):
    raise ImportError('verifyparams requires Python 3.9 or above.')
del sys

__version__ = '0.0.1'
__author__ = 'John Indika'
__credits__ = 'STEM Research'
__email__ = "verifyparams@stemfard.org"
__description__ = "Parameter validation library for STEM and data-intensive applications"
__url__ = "https://verifyparams.stemfard.org"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2026 {__credits__}"

from .verifiers import (
    # arraylike
    verify_elements_in_range,
    verify_all_integers,
    verify_all_positive,
    verify_len_equal,
    verify_diff_constant,
    verify_strictly_increasing,
    verify_distinct,
    verify_not_constant,
    verify_lower_lte_upper_arr,
    verify_numeric_arr,
    
    # dataframes
    verify_axis,
    verify_dframe,
    verify_series,
    verify_df_columns,
    
    # dtypes
    verify_boolean,
    verify_int,
    verify_float,
    verify_int_or_float,
    verify_finite,
    verify_positive,
    verify_complex,
    
    # func_args
    verify_args_count,

    # function
    verify_function,
    
    # intervals
    verify_lower_lte_upper,
    verify_step_size,
    
    # linalgebra
    verify_array_or_matrix,
    verify_square,
    verify_linear_system,
    
    # membership
    verify_membership,
    verify_membership_allowed,
    
    # numeric
    verify_decimals,
    verify_numeric,
    
    # stats
    verify_sta_conf_level,
    verify_sta_sig_level,
    verify_sta_alternative,
    verify_sta_decision,
    
    # strings
    verify_string,
    verify_str_email,
    verify_str_identifier,
    verify_str_alphanumeric,
    verify_str_numeric,
    
    # symbolic
    verify_symbolic_expr
)

__all__ = [
    # arraylike
    "verify_elements_in_range",
    "verify_all_integers",
    "verify_all_positive",
    "verify_len_equal",
    "verify_diff_constant",
    "verify_strictly_increasing",
    "verify_distinct",
    "verify_not_constant",
    "verify_lower_lte_upper_arr",
    "verify_numeric_arr",

    # dataframes
    "verify_axis",
    "verify_dframe",
    "verify_series",
    "verify_df_columns",

    # dtypes
    "verify_boolean",
    "verify_int",
    "verify_float",
    "verify_int_or_float",
    "verify_complex",
    "verify_positive",
    "verify_finite",
    
    # func_args
    "verify_args_count",
    
    # functions
    "verify_function",

    # intervals
    "verify_lower_lte_upper",
    "verify_step_size",

    # linalgebra
    "verify_array_or_matrix",
    "verify_square",
    "verify_linear_system",

    # membership
    "verify_membership",
    "verify_membership_allowed",

    # numeric
    "verify_decimals",
    "verify_numeric",

    # stats
    "verify_sta_conf_level",
    "verify_sta_sig_level",
    "verify_sta_alternative",
    "verify_sta_decision",

    # strings
    "verify_string",
    "verify_str_email",
    "verify_str_identifier",
    "verify_str_alphanumeric",
    "verify_str_numeric",
    
    # symbolic
    "verify_symbolic_expr"
]

#==============================================================================#
#                                                                              #
# STEM RESEARCH :: AI . APIs . Innovation :: https://verifyparams.stemfard.org #
#                                                                              #
#==============================================================================#