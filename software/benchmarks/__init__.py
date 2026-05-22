"""Shared benchmark tasks for MRU experiments.

Available task modules:
  mnist    — sequential and permuted MNIST
  lra      — Long Range Arena (ListOps, IMDb byte-level)
  audio    — Google Speech Commands (binary, digits, all-30)

The central BENCHMARK_TASKS dict is available directly from this package.
"""

from .registry import BENCHMARK_TASKS

__all__ = ["BENCHMARK_TASKS"]
