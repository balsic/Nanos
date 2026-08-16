"""Narrow, opt-in shims for upstream bugs that block dataset preparation.

Rules for anything added here:

1. It fixes a defect in a *third-party* package, not in this one.
2. It is the smallest change that unblocks, and it changes no behaviour on the path that
   already works.
3. It states which versions it was needed for, so it can be deleted rather than
   accumulating forever.
4. It is applied explicitly by the code that needs it, never as an import side effect --
   a monkey-patch that installs itself on import is indistinguishable from the library
   behaving that way, which is how a workaround becomes a permanent mystery.
"""

from __future__ import annotations

import warnings

__all__ = ["patch_pymatreader_opaque"]

_PATCHED = {"pymatreader": False}


def patch_pymatreader_opaque() -> bool:
    """Make ``pymatreader`` tolerate array-valued MatlabOpaque fields.

    Needed for: EEGLAB ``.set`` files in GrosseWentrup2009 (and any corpus saved the same
    way), with pymatreader 1.2.3 + scipy >= 1.15.

    The bug: ``_check_for_scipy_mat_struct`` tests ``data[0][2] == b'string'`` to detect a
    Matlab string object. Newer scipy returns an *array* there, so the comparison yields
    an array and ``if`` raises ``ValueError: The truth value of an array with more than
    one element is ambiguous``. The guard catches only ``IndexError``, so the exception
    escapes and the file cannot be read at all.

    The fix reduces the comparison with ``np.all`` and widens the guard. Semantics on the
    scalar path are unchanged: for a scalar, ``np.all(x == b'string')`` is exactly
    ``x == b'string'``.

    pymatreader 1.2.3 is the newest release, so upgrading does not help; this stays until
    upstream moves. Returns ``True`` if the patch was applied or already in place.
    """
    if _PATCHED["pymatreader"]:
        return True
    try:
        import numpy as np
        from pymatreader import utils as _u
        from scipy.io.matlab import MatlabOpaque
    except ImportError:
        return False

    original = _u._check_for_scipy_mat_struct

    def patched(data):
        if isinstance(data, MatlabOpaque):
            # The MatlabOpaque branch is handled here in full rather than delegated.
            # Delegating after the guard looks tidier and does not work: the original
            # would simply re-execute the same broken comparison and raise anyway. The
            # branch is short and its remainder is reproduced verbatim below.
            try:
                # np.all() reduces the array-valued comparison newer scipy produces; on a
                # scalar this is identical to the original test.
                if np.all(data[0][2] == b"string"):
                    warnings.warn(
                        "pymatreader cannot import Matlab string variables; "
                        "convert them to char arrays in Matlab.",
                        RuntimeWarning, stacklevel=2,
                    )
                    return None
            except (IndexError, ValueError):
                pass
            warnings.warn(
                "Complex objects (like classes) are not supported. They are imported "
                "on a best effort basis but your mileage will vary.",
                RuntimeWarning, stacklevel=2,
            )
            # Fall through exactly as upstream does: an opaque object may still be an
            # ndarray that the downstream handlers can partially recover.
            if isinstance(data, np.ndarray):
                return _u._handle_scipy_ndarray(data)
            return data
        return original(data)

    # Rebound on the module so the function's own recursive calls -- which resolve through
    # module globals -- also take the patched path. Patching only the imported reference
    # would leave every nested struct going through the broken version.
    _u._check_for_scipy_mat_struct = patched
    _PATCHED["pymatreader"] = True
    return True
