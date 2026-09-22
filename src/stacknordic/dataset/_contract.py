"""Self-describing HDF5 dataset contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import h5py

SNAPSHOT_SCHEMA = "stacknordic-hdf5-snapshot-v1"
SERIES_SCHEMA = "stacknordic-hdf5-series-v1"
CONTRACT_ATTRIBUTE = "stacknordic_contract"
SNAPSHOTS_ATTRIBUTE = "stacknordic_snapshots"


def normalise_hdf5_path(value: str) -> str:
    """Normalise one relative HDF5 path without accepting traversal."""

    if not isinstance(value, str):
        raise TypeError("field output paths must be strings")
    parts = tuple(part.strip() for part in value.split("/"))
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"invalid HDF5 field path: {value!r}")
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class HDF5Contract:
    """Immutable logical-field contract stored with every package."""

    schema: str
    name: str
    dtype: str
    shape_zyx: tuple[int, int, int]
    fields: MappingProxyType
    components: MappingProxyType
    metadata: MappingProxyType

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "dtype": self.dtype,
            "shape_zyx": list(self.shape_zyx),
            "fields": dict(self.fields),
            "components": {
                name: list(values) for name, values in self.components.items()
            },
            "grid": {
                "x": "grid/x_axis",
                "y": "grid/y_axis",
                "z": "grid/z_axis",
            },
            "metadata": dict(self.metadata),
        }

    def with_schema(self, schema: str) -> HDF5Contract:
        return HDF5Contract(
            schema=schema,
            name=self.name,
            dtype=self.dtype,
            shape_zyx=self.shape_zyx,
            fields=self.fields,
            components=self.components,
            metadata=self.metadata,
        )


def contract(
    *,
    schema: str,
    name: str,
    dtype: str,
    shape_zyx: tuple[int, int, int],
    fields: dict[str, str],
    components: dict[str, tuple[str, ...]],
    metadata: dict[str, object],
) -> HDF5Contract:
    if schema not in {SNAPSHOT_SCHEMA, SERIES_SCHEMA}:
        raise ValueError(f"unsupported StackNordic HDF5 schema: {schema!r}")
    if set(fields) != set(components):
        raise ValueError("HDF5 fields and component mappings must have identical keys")
    return HDF5Contract(
        schema=schema,
        name=name,
        dtype=dtype,
        shape_zyx=tuple(int(value) for value in shape_zyx),
        fields=MappingProxyType(dict(fields)),
        components=MappingProxyType(
            {key: tuple(values) for key, values in components.items()}
        ),
        metadata=MappingProxyType(dict(metadata)),
    )


def write_contract(file: h5py.File, definition: HDF5Contract) -> None:
    file.attrs[CONTRACT_ATTRIBUTE] = json.dumps(
        definition.payload(), ensure_ascii=False, separators=(",", ":")
    )


def read_contract(file: h5py.File) -> HDF5Contract:
    raw = file.attrs.get(CONTRACT_ATTRIBUTE)
    if raw is None:
        raise ValueError("HDF5 file does not contain a StackNordic contract")
    try:
        payload = json.loads(str(raw))
        return contract(
            schema=str(payload["schema"]),
            name=str(payload["name"]),
            dtype=str(payload["dtype"]),
            shape_zyx=tuple(payload["shape_zyx"]),
            fields={
                str(name): normalise_hdf5_path(path)
                for name, path in payload["fields"].items()
            },
            components={
                str(name): tuple(str(component) for component in values)
                for name, values in payload["components"].items()
            },
            metadata=dict(payload.get("metadata") or {}),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("HDF5 file contains an invalid StackNordic contract") from error


def compatible(first: HDF5Contract, second: HDF5Contract) -> bool:
    return (
        first.name == second.name
        and first.dtype == second.dtype
        and first.shape_zyx == second.shape_zyx
        and dict(first.fields) == dict(second.fields)
        and dict(first.components) == dict(second.components)
    )
