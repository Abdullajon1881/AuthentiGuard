"""AuthentiGuard text detector. Import from `pipeline` for all inference."""

from .pipeline import (
    MODEL_VERSION,
    TextDetector,
    analyze,
    analyze_to_dict,
    build_detector,
)

__all__ = [
    "analyze",
    "analyze_to_dict",
    "build_detector",
    "TextDetector",
    "MODEL_VERSION",
]
