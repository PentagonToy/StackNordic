"""Shared progress-display lifecycle."""

from __future__ import annotations

import onsaemiro as osm


def start_progress(total: int, description: str):
    """Create and immediately render a progress display."""

    progress_type = getattr(osm, "Progress", None) or osm.ProgressBar
    reporter = progress_type(total=total, desc=description)
    reporter.update(0)
    return reporter


def finish_progress(reporter) -> None:
    """Preserve the final display across supported Onsaemiro versions."""

    finish = getattr(reporter, "finish", None)
    finish() if finish is not None else reporter.close()
