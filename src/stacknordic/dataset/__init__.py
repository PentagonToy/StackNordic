"""Dataset preparation operations."""

from .compute import ComputedDataset, compute
from .downsample import DownsampledDataset, downsample
from .export import ExportedDataset, export
from .package import CompiledDataset, PackagedDataset, compile, package
from .postprocess import PostprocessedDataset, postprocess


class Dataset:
    """Dataset creation and reduction operations."""

    export = staticmethod(export)
    compute = staticmethod(compute)
    downsample = staticmethod(downsample)
    postprocess = staticmethod(postprocess)
    package = staticmethod(package)
    compile = staticmethod(compile)


__all__ = [
    "CompiledDataset",
    "ComputedDataset",
    "Dataset",
    "DownsampledDataset",
    "ExportedDataset",
    "PackagedDataset",
    "PostprocessedDataset",
    "compile",
    "compute",
    "downsample",
    "export",
    "package",
    "postprocess",
]


def __dir__() -> list[str]:
    return sorted(__all__)
