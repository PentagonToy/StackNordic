"""OpenFOAM reconstruction and post-processing."""

from . import openfoam
from .computing import Computing, ProgressComputation, ReactionRateComputation
from .dataset import (
    CompiledDataset,
    ComputedDataset,
    Dataset,
    DownsampledDataset,
    ExportedDataset,
    PackagedDataset,
    PostprocessedDataset,
)
from .filtering import Filtering
from .openfoam.postprocess import PostprocessedCase
from .openfoam.write import ComputedFields
from .postprocess import Postprocess, PostprocessOperation
from .reconstruction import Reconstruction, reconstruct

__version__ = "0.1.0"

OpenFOAM = openfoam

__all__ = [
    "CompiledDataset",
    "ComputedDataset",
    "ComputedFields",
    "Computing",
    "Dataset",
    "DownsampledDataset",
    "ExportedDataset",
    "Filtering",
    "OpenFOAM",
    "PackagedDataset",
    "Postprocess",
    "PostprocessOperation",
    "PostprocessedCase",
    "PostprocessedDataset",
    "ProgressComputation",
    "ReactionRateComputation",
    "Reconstruction",
    "reconstruct",
]


def __dir__() -> list[str]:
    return sorted(__all__)
