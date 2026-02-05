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

from .verifiers import *

#==============================================================================#
#                                                                              #
# STEM RESEARCH :: AI . APIs . Innovation :: https://verifyparams.stemfard.org #
#                                                                              #
#==============================================================================#