"""Shared progress-display lifecycle."""

from __future__ import annotations

import hedgehogs as hdg


def start_progress(total: int, description: str):
    """Create and immediately render a progress display."""

    reporter = hdg.Progress(total=total, desc=description)
    reporter.update(0)
    return reporter


def finish_progress(reporter) -> None:
    """Preserve the final Hedgehogs progress display."""

    reporter.finish()
