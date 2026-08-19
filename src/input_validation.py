"""Input normalization shared by the valuation UI."""

from __future__ import annotations

import math
from numbers import Real


def normalize_whole_number(
    value: Real,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Round half up to a whole number and clamp it to an allowed range."""

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("A finite numeric value is required.")
    if minimum > maximum:
        raise ValueError("The minimum cannot exceed the maximum.")

    rounded_value = math.floor(numeric_value + 0.5)
    return max(minimum, min(maximum, rounded_value))
