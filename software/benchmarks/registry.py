"""Central benchmark task registry.

Merges all per-task-type task dicts into the single BENCHMARK_TASKS dict.
"""

from benchmarks.mnist import MNIST_TASKS
from benchmarks.lra import LRA_TASKS
from benchmarks.audio import AUDIO_TASKS
from benchmarks.shakespeare import SHAKESPEARE_TASKS, FULL_SHAKESPEARE_TASKS

BENCHMARK_TASKS = {
    **MNIST_TASKS,
    **LRA_TASKS,
    **AUDIO_TASKS,
    **SHAKESPEARE_TASKS,
    **FULL_SHAKESPEARE_TASKS,
}
