"""Benchmarking and performance analysis suite."""

from .benchmark import (
    run_variant_benchmark,
    compare_model_sizes,
    DEFAULT_SEARCH_VARIANTS,
)

__all__ = [
    "run_variant_benchmark",
    "compare_model_sizes",
    "DEFAULT_SEARCH_VARIANTS",
]
