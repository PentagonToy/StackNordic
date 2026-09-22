"""Decode stored OpenFOAM fields and collated processor blocks."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class OpenFOAMDecodeError(ValueError):
    """Raised when stored OpenFOAM data cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class FieldData:
    field_class: str
    values: np.ndarray


_COMMENTS = re.compile(rb"/\*.*?\*/|//[^\n]*", re.DOTALL)
_COMPONENTS = {
    b"scalar": 1,
    b"sphericalTensor": 1,
    b"vector": 3,
    b"symmTensor": 6,
    b"tensor": 9,
}
_NONUNIFORM = re.compile(
    rb"\b(?:internalField\s+)?nonuniform\s+"
    rb"List<(scalar|sphericalTensor|vector|symmTensor|tensor)>\s+(\d+)\s*\("
)


def _block(payload: bytes, name: bytes) -> bytes | None:
    match = re.search(rb"\b" + re.escape(name) + rb"\s*\{", payload)
    if match is None:
        return None
    depth = 1
    for index in range(match.end(), len(payload)):
        if payload[index] == ord("{"):
            depth += 1
        elif payload[index] == ord("}"):
            depth -= 1
            if depth == 0:
                return payload[match.end() : index]
    raise OpenFOAMDecodeError(f"unterminated {name.decode()} dictionary")


def _block_end(payload: bytes, name: bytes) -> int:
    match = re.search(rb"\b" + re.escape(name) + rb"\s*\{", payload)
    if match is None:
        raise OpenFOAMDecodeError(f"data is missing its {name.decode()} dictionary")
    depth = 1
    for index in range(match.end(), len(payload)):
        if payload[index] == ord("{"):
            depth += 1
        elif payload[index] == ord("}"):
            depth -= 1
            if depth == 0:
                return index + 1
    raise OpenFOAMDecodeError(f"unterminated {name.decode()} dictionary")


def _entry(block: bytes, name: bytes) -> str | None:
    match = re.search(rb"\b" + re.escape(name) + rb'\s+(?:"([^"]*)"|([^;]+));', block)
    if match is None:
        return None
    value = match.group(1) if match.group(1) is not None else match.group(2)
    return value.decode().strip()


def _header(payload: bytes) -> tuple[str, str, str | None]:
    header = _block(_COMMENTS.sub(b" ", payload), b"FoamFile")
    if header is None:
        raise OpenFOAMDecodeError("data is missing its FoamFile dictionary")
    field_class = _entry(header, b"class")
    if not field_class:
        raise OpenFOAMDecodeError("data is missing its OpenFOAM class")
    return field_class, _entry(header, b"format") or "ascii", _entry(header, b"arch")


def _binary_dtypes(arch: str | None, kind: str) -> tuple[np.dtype, ...]:
    code = "i" if kind == "label" else "f"
    if arch is None:
        widths = (8, 4)
        byte_order = "<"
    else:
        byte_order = "<" if arch.startswith("LSB;") else ">" if arch.startswith("MSB;") else "="
        match = re.search(rf"(?:^|;){kind}=(32|64)(?:;|$)", arch)
        widths = (int(match.group(1)) // 8,) if match is not None else (8, 4)
    return tuple(np.dtype(f"{byte_order}{code}{width}") for width in widths)


def _binary_values(
    payload: bytes,
    start: int,
    count: int,
    components: int,
    arch: str | None,
    kind: str,
) -> np.ndarray:
    starts = (start,)
    if payload[start : start + 2] == b"\r\n":
        starts = (start + 2, start)
    elif payload[start : start + 1] == b"\n":
        starts = (start + 1, start)
    for dtype in _binary_dtypes(arch, kind):
        size = count * components * dtype.itemsize
        for data_start in starts:
            data_end = data_start + size
            if re.match(rb"\s*\)\s*;?", payload[data_end:]) is not None:
                return np.frombuffer(payload[data_start:data_end], dtype=dtype).astype(
                    dtype.newbyteorder("="), copy=True
                )
    raise OpenFOAMDecodeError("binary list byte count does not match its declaration")


def read_field_payload(payload: bytes, *, cell_count: int | None = None) -> FieldData:
    """Decode one complete OpenFOAM volume-field payload."""

    field_class, format_name, arch = _header(payload)
    uniform = re.search(rb"\binternalField\s+uniform\s+([^;]+);", payload, re.DOTALL)
    if uniform is not None:
        values = np.fromstring(
            uniform.group(1).replace(b"(", b" ").replace(b")", b" "), sep=" "
        )
        if values.size not in {1, 3, 6, 9}:
            raise OpenFOAMDecodeError("uniform internalField has an invalid component count")
        if cell_count is None:
            raise OpenFOAMDecodeError("uniform internalField requires its local cell count")
        values = np.broadcast_to(values, (cell_count, values.size)).copy()
        if values.shape[1] == 1:
            values = values[:, 0]
        return FieldData(field_class, values)
    match = re.search(rb"\binternalField\s+" + _NONUNIFORM.pattern, payload)
    if match is None:
        raise OpenFOAMDecodeError("could not locate a supported internalField")
    kind, raw_count = match.groups()
    count = int(raw_count)
    components = _COMPONENTS[kind]
    if format_name == "binary":
        values = _binary_values(payload, match.end(), count, components, arch, "scalar")
    else:
        end = re.search(rb"\)\s*;", payload[match.end() :], re.DOTALL)
        if end is None:
            raise OpenFOAMDecodeError("ASCII internalField is unterminated")
        raw = payload[match.end() : match.end() + end.start()]
        values = np.fromstring(raw.replace(b"(", b" ").replace(b")", b" "), sep=" ")
        if values.size != count * components:
            raise OpenFOAMDecodeError("ASCII internalField count does not match its declaration")
    if components > 1:
        values = values.reshape(count, components)
    return FieldData(field_class, values)


def read_label_payload(payload: bytes) -> np.ndarray:
    """Decode an OpenFOAM label list."""

    _, format_name, arch = _header(payload)
    header_end = _block_end(payload, b"FoamFile")
    trailing = _COMMENTS.sub(lambda match: b" " * len(match.group()), payload[header_end:])
    match = re.search(rb"\b(\d+)\s*\(", trailing)
    if match is None:
        raise OpenFOAMDecodeError("could not locate label-list contents")
    count = int(match.group(1))
    start = header_end + match.end()
    if format_name == "binary":
        values = _binary_values(payload, start, count, 1, arch, "label")
    else:
        end = re.search(rb"\)\s*;?", payload[start:], re.DOTALL)
        if end is None:
            raise OpenFOAMDecodeError("ASCII label list is unterminated")
        values = np.fromstring(payload[start : start + end.start()], sep=" ", dtype=np.int64)
        if values.size != count:
            raise OpenFOAMDecodeError("ASCII label-list count does not match its declaration")
    return values.astype(np.int64, copy=False)


def collated_blocks(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield processor number and exact local payload from decomposedBlockData."""

    marker = re.compile(rb"\s*//\s*Processor(\d+)\s*")
    seen: set[int] = set()
    local_header: bytes | None = None
    with path.open("rb") as stream:
        preamble = bytearray()
        first_marker: tuple[int, bytes] | None = None
        while line := stream.readline():
            match = marker.fullmatch(line)
            if match is not None:
                first_marker = (int(match.group(1)), line)
                break
            preamble.extend(line)
        field_class, _, _ = _header(bytes(preamble))
        if field_class != "decomposedBlockData":
            raise OpenFOAMDecodeError(f"{path} is not collated decomposedBlockData")
        pending = first_marker
        while pending is not None:
            processor = pending[0]
            size_line = stream.readline()
            while size_line and not size_line.strip():
                size_line = stream.readline()
            try:
                size = int(size_line.strip())
            except ValueError as error:
                raise OpenFOAMDecodeError(
                    f"collated block for processor{processor} has no byte count"
                ) from error
            opening = stream.read(1)
            while opening and opening.isspace():
                opening = stream.read(1)
            if opening != b"(":
                raise OpenFOAMDecodeError(
                    f"collated block for processor{processor} has no opening delimiter"
                )
            block = stream.read(size)
            if len(block) != size or stream.read(1) != b")":
                raise OpenFOAMDecodeError(
                    f"collated block for processor{processor} is truncated"
                )
            if processor in seen:
                raise OpenFOAMDecodeError(f"collated data repeats processor{processor}")
            seen.add(processor)
            header_start = re.search(rb"\bFoamFile\s*\{", block)
            if header_start is not None:
                local_header = block[header_start.start() : _block_end(block, b"FoamFile")]
            elif local_header is not None:
                block = local_header + b"\n" + block
            else:
                raise OpenFOAMDecodeError(
                    f"first collated block in {path} does not contain its local FoamFile header"
                )
            yield processor, block
            pending = None
            while line := stream.readline():
                match = marker.fullmatch(line)
                if match is not None:
                    pending = (int(match.group(1)), line)
                    break
    if not seen:
        raise OpenFOAMDecodeError(f"{path} contains no processor blocks")


def read_field(path: Path, *, cell_count: int | None = None) -> FieldData:
    return read_field_payload(path.read_bytes(), cell_count=cell_count)


def read_labels(path: Path) -> np.ndarray:
    return read_label_payload(path.read_bytes())
