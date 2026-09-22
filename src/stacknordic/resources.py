"""Shared local execution-resource policy."""

from __future__ import annotations

import os
from pathlib import Path


def available_cpus() -> int:
    """Return the smallest CPU allowance visible to the current process."""

    limits = [os.cpu_count() or 1]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except AttributeError:
        pass
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        try:
            value = int(os.environ.get(name, ""))
        except ValueError:
            continue
        if value > 0:
            limits.append(value)
    return max(1, min(limits))


def workers(n_jobs: int | None, work_count: int) -> int:
    """Resolve a requested worker count against available independent work."""

    if n_jobs is None:
        return 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1, a positive integer, or None")
    if n_jobs == -1:
        return min(work_count, available_cpus())
    return min(work_count, n_jobs)


def available_memory() -> int | None:
    """Return the smallest observable process or host memory allowance."""

    limits: list[int] = []
    paths = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                current = Path("/sys/fs/cgroup") / relative.lstrip("/")
                while True:
                    paths.insert(0, current / "memory.max")
                    if current == Path("/sys/fs/cgroup"):
                        break
                    current = current.parent
                break
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    for path in paths:
        try:
            value = path.read_text().strip()
            if value != "max":
                limits.append(int(value))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    try:
        limits.append(int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")))
    except (AttributeError, OSError, ValueError):
        pass
    return min(limits) if limits else None


def memory_bounded_workers(
    n_jobs: int | None,
    work_count: int,
    estimated_worker_bytes: int,
) -> int:
    """Resolve workers and constrain automatic execution by available memory."""

    selected = workers(n_jobs, work_count)
    if n_jobs != -1:
        return selected
    available = available_memory()
    if available is None:
        return selected
    safe = max(1, int(available * 0.70) // max(estimated_worker_bytes, 1))
    return min(selected, safe)
