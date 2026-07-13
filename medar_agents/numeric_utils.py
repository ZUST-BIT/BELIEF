"""Numeric normalization helpers for model-produced values."""

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float, falling back for invalid or non-finite values."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def clamp_confidence(value: Any, default: float = 0.0) -> float:
    """Normalize a confidence-like value to the inclusive unit interval."""
    number = safe_float(value, default=default)
    return max(0.0, min(1.0, number))
