import abc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from joblib.memory import Memory
from torch.utils.data import DataLoader
from torchvision.io import encode_jpeg
from torchvision.transforms.v2.functional import convert_image_dtype
from tqdm import tqdm

from aeroblade.data import ImageFolder
from aeroblade.image import extract_patches

from meaningful_complexity.measure_complexity import ComplexityMeasurer


mem = Memory(location="cache", compress=("lz4", 9), verbose=0)


class Complexity(abc.ABC):
    """Base class for all complexity metrics."""

    @torch.no_grad()
    def compute(self, ds: ImageFolder) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        Compute complexity of dataset.
        """

        files = [Path(f).name for f in ds.img_paths]
        result = self._compute(ds=ds)
        return self._postprocess(result), files

    @abc.abstractmethod
    def _compute(self, ds: ImageFolder) -> Any:
        """Metric-specific computation."""
        pass

    @abc.abstractmethod
    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        """Post-processing step, that maps result into dictionary."""
        pass


@mem.cache(ignore=["num_workers"])
def _compute_jpeg(
    ds: ImageFolder, quality: int, patch_size: int, patch_stride: int, num_workers: int
) -> torch.Tensor:
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers)

    image_results = []
    for tensor, _ in tqdm(dl, desc="Computing JPEG complexity", total=len(dl)):
        if patch_size is None:
            patches = [tensor[0]]
        else:
            patches = extract_patches(
                array=tensor, size=patch_size, stride=patch_stride
            )[0]

        patch_results = []
        for patch in patches:
            nbytes = len(
                encode_jpeg(convert_image_dtype(patch, torch.uint8), quality=quality)
            )
            patch_results.append(nbytes)
        image_results.append(torch.tensor(patch_results, dtype=torch.float16))
    return torch.stack(image_results) / (patch.shape[1] * patch.shape[2])  # normalize


class JPEG(Complexity):
    def __init__(
        self,
        quality: int = 50,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        num_workers: int = 0,
    ) -> None:
        """
        quality: JPEG quality to use
        """
        self.quality = quality
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_workers = num_workers

    def _compute(self, ds: ImageFolder) -> Any:
        return _compute_jpeg(
            ds=ds,
            quality=self.quality,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            num_workers=self.num_workers,
        )

    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        return {f"jpeg_{self.quality}": result}


# Wrap the meaningful complexity interpret method with caching
@mem.cache
def cached_meaningful_interpret(comp_meas_params, patch_np):
    comp_meas = ComplexityMeasurer(**comp_meas_params)

    return comp_meas.interpret(patch_np)


@mem.cache(ignore=["num_workers"])
def _compute_meaningful(
    ds: ImageFolder, comp_meas_params: dict, patch_size: int, patch_stride: int, num_workers: int
) -> torch.Tensor:
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers)

    image_results = []

    for tensor, _ in tqdm(dl, desc="Computing Meaningful complexity", total=len(dl)):
        if patch_size is None:
            patches = [tensor[0]]
        else:
            patches = extract_patches(
                array=tensor, size=patch_size, stride=patch_stride
            )[0]

        patch_results = []

        for patch in patches:
            patch_np = patch.squeeze().numpy()

            # Check if the patch is uniform
            if patch_np.min() == patch_np.max():
                complexity = 0
            else:
                complexity = cached_meaningful_interpret(comp_meas_params, patch_np)

            patch_results.append(np.sum(complexity))

        image_results.append(torch.tensor(patch_results, dtype=torch.float16))

    return torch.stack(image_results) / (patch.shape[1] * patch.shape[2])  # normalize


class Meaningful(Complexity):
    def __init__(
        self,
        comp_meas_params: dict,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        num_workers: int = 0,
    ) -> None:
        """
        comp_meas_params: Parameters for the ComplexityMeasurer.
        """
        self.comp_meas_params = comp_meas_params
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_workers = num_workers

    def _compute(self, ds: ImageFolder) -> Any:
        return _compute_meaningful(
            ds=ds,
            comp_meas_params=self.comp_meas_params,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            num_workers=self.num_workers,
        )

    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        return {"meaningful": result}


def complexity_from_config(
    config: str, patch_size: int, patch_stride: int, batch_size: int, num_workers: int
) -> Complexity:
    """Parse config string and return matching complexity metric."""
    if config.startswith("jpeg"):
        _, quality = config.split("_")
        return JPEG(
            quality=int(quality),
            patch_size=patch_size,
            patch_stride=patch_stride,
            num_workers=num_workers,
        )
    elif config == "meaningful":
        comp_meas_params = {
            "ncs_to_check": 8,
            "n_cluster_inits": 1,
            "nz": 2,
            "num_levels": 4,
            "cluster_model": "GMM",
            "info_subsample": 0.3,
            "suppress_all_prints": True
        }

        return Meaningful(
            comp_meas_params=comp_meas_params,
            patch_size=patch_size,
            patch_stride=patch_stride,
            num_workers=num_workers,
        )
    else:
        raise NotImplementedError(f"No matching complexity metric for {config}.")
