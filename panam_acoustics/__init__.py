"""Local minimal acoustics replacements (BSD-licensed portions)."""

from .atmosphere import Atmosphere
from .filters import highpass, lowpass

__all__ = [
    "Atmosphere",
    "highpass",
    "lowpass",
]
