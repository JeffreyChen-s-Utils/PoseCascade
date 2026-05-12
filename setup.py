"""Build hook for Cython kernels.

``pyproject.toml`` carries the canonical project metadata; this script
exists only because ``setuptools.build_meta`` cannot run ``cythonize`` on
its own. ``pip install -e .`` (or ``pip install .``) walks through here,
cythonizes every ``.pyx`` listed below, and builds the resulting C
sources into shared objects alongside the source tree.

The build is intentionally *optional* — if cythonize fails (no compiler,
Cython missing, Python ABI mismatch) the resulting wheel just doesn't ship
the extension, and the Python wrapper in :mod:`posecascade.animation.cloth`
falls back to the NumPy implementation. That keeps source checkouts
runnable without forcing every contributor to install a C toolchain.
"""
from __future__ import annotations

import sys

import numpy as np
from setuptools import setup

# Lazy Cython import so a Cython-less environment still gets a working
# wheel (sdist install + no ``cython`` in build deps). Empty
# ``ext_modules`` list = pure-Python install.
ext_modules: list = []
try:
    from Cython.Build import cythonize
    from setuptools import Extension
except ImportError:
    sys.stderr.write(
        "Cython not found — building without the cloth-solver C kernels. "
        "The Python NumPy fallback will be used at runtime.\n",
    )
else:
    ext_modules = cythonize(
        [
            Extension(
                "posecascade.animation._cloth_kernels",
                ["posecascade/animation/_cloth_kernels.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
        ],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    )

setup(ext_modules=ext_modules)
