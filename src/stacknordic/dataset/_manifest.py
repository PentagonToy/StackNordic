"""Dataset manifest compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping


def component_file(record: Mapping[str, object], component: str) -> str:
    """Return a component path from current or legacy export manifests."""

    key = f"{component} filename"
    if key in record:
        return str(record[key])
    if component in record:
        return str(record[component])
    raise KeyError(key)
