"""Platform workaround for spurious BLAS floating-point flags.

Apple's Accelerate framework leaves divide-by-zero / overflow / invalid flags
set after certain matmuls, which NumPy then reports as RuntimeWarning on every
later call even though the values are exact (a plain random ``A @ A`` triggers
it).  We suppress only that platform quirk, only inside our own kernels; on
other platforms NumPy's error reporting is untouched.
"""
from __future__ import annotations

import functools
import sys

import numpy as np


def quiet_blas(fn):
    if sys.platform != "darwin":
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with np.errstate(all="ignore"):
            return fn(*args, **kwargs)

    return wrapper
