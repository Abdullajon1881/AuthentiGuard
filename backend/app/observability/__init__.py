"""Observability package — Stage 3 monitoring + drift detection.

Everything under this package is OFF THE HOT PATH. Any failure in a
logger, sampler, or metric writer must NEVER affect the inference
response. Modules here are imported lazily from callers and their
public functions all swallow exceptions with a structlog warning.
"""
