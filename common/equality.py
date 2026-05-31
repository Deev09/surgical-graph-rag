"""Deep equality for bundles containing numpy arrays.

Default dataclass equality on np.ndarray fields raises or behaves
ambiguously. This module walks frozen dataclasses recursively and uses an
explicit array comparator that checks dtype + shape + values.

Tuple-vs-list is treated as significant: a field that was a tuple at dump
time MUST be a tuple after load. Same for list. This catches serde bugs
that flatten everything to list.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def array_aware_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    a_is_arr = isinstance(a, np.ndarray)
    b_is_arr = isinstance(b, np.ndarray)
    if a_is_arr or b_is_arr:
        if not (a_is_arr and b_is_arr):
            return False
        if a.dtype != b.dtype:
            return False
        if a.shape != b.shape:
            return False
        return bool(np.array_equal(a, b))

    a_is_dc = is_dataclass(a) and not isinstance(a, type)
    b_is_dc = is_dataclass(b) and not isinstance(b, type)
    if a_is_dc or b_is_dc:
        if not (a_is_dc and b_is_dc):
            return False
        if type(a) is not type(b):
            return False
        for f in fields(a):
            if not array_aware_equal(getattr(a, f.name), getattr(b, f.name)):
                return False
        return True

    if isinstance(a, tuple) or isinstance(b, tuple):
        if not (isinstance(a, tuple) and isinstance(b, tuple)):
            return False
        if len(a) != len(b):
            return False
        return all(array_aware_equal(x, y) for x, y in zip(a, b))

    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            return False
        if len(a) != len(b):
            return False
        return all(array_aware_equal(x, y) for x, y in zip(a, b))

    if isinstance(a, (set, frozenset)) or isinstance(b, (set, frozenset)):
        if type(a) is not type(b):
            return False
        return a == b

    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        if set(a.keys()) != set(b.keys()):
            return False
        return all(array_aware_equal(a[k], b[k]) for k in a)

    return a == b
