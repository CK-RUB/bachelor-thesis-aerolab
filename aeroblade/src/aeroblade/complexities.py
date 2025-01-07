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

import cv2
from scipy.ndimage import convolve

from aeroblade.data import ImageFolder
from aeroblade.image import extract_patches

from aeroblade.external.meaningful_complexity import MeaningfulComplexity


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

    for tensor, _ in tqdm(dl, desc=f"Computing JPEG_{quality} complexity", total=len(dl)):
        if patch_size is None:
            patches = [tensor[0]]
        else:
            patches = extract_patches(
                array=tensor, size=patch_size, stride=patch_stride
            )[0]

        patch_results = []

        for patch in patches:
            nbytes = len(encode_jpeg(convert_image_dtype(patch, torch.uint8), quality=quality))
            patch_results.append(nbytes)

        image_results.append(torch.tensor(patch_results, dtype=torch.float64))

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


@mem.cache
def calculate_pixel_variance(image, neighborhood_size):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel = np.ones((neighborhood_size, neighborhood_size))
    kernel /= kernel.size
    mean = convolve(gray_image, kernel)
    mean_sq = convolve(gray_image ** 2, kernel)

    # var formula: variance = E[X^2] - (E[X])^2
    variance_map = mean_sq - mean ** 2

    return variance_map


@mem.cache(ignore=["num_workers"])
def _compute_variance(
    ds: ImageFolder, neighborhood_size: int, patch_size: int, patch_stride: int, num_workers: int
) -> torch.Tensor:
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers)

    image_results = []

    for tensor, _ in tqdm(dl, desc=f"Computing VARIANCE_{neighborhood_size} complexity", total=len(dl)):
        if patch_size is None:
            patches = [tensor[0]]
        else:
            patches = extract_patches(
                array=tensor, size=patch_size, stride=patch_stride
            )[0]

        patch_results = []

        for patch in patches:
            patch_np = patch.permute(1, 2, 0).numpy()  # Convert to HWC format for OpenCV
            variance_map = calculate_pixel_variance(patch_np, neighborhood_size)
            patch_variance = np.mean(variance_map)  # Aggregate variance for the patch
            patch_results.append(patch_variance)

        image_results.append(torch.tensor(patch_results, dtype=torch.float64))

    return torch.stack(image_results) / (patch.shape[1] * patch.shape[2])  # normalize


class Variance(Complexity):
    def __init__(
        self,
        neighborhood_size: int = 8,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        num_workers: int = 0,
    ) -> None:
        """
        Variance complexity metric with neighborhood variance.
        """
        self.neighborhood_size = neighborhood_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_workers = num_workers

    def _compute(self, ds: ImageFolder) -> Any:
        return _compute_variance(
            ds=ds,
            neighborhood_size=self.neighborhood_size,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            num_workers=self.num_workers,
        )

    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        return {f"variance_{self.neighborhood_size}": result}


# Wrap the meaningful complexity interpret method with caching
@mem.cache
def cached_meaningful_interpret(meaningful_params, patch_np):
    comp_meas = MeaningfulComplexity(**meaningful_params)

    return comp_meas.interpret(patch_np)


@mem.cache(ignore=["num_workers"])
def _compute_meaningful(
    ds: ImageFolder, meaningful_params: dict, patch_size: int, patch_stride: int, num_workers: int
) -> torch.Tensor:
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers)

    image_results = []

    for tensor, _ in tqdm(dl, desc="Computing MEANINGFUL complexity", total=len(dl)):
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
                complexity = cached_meaningful_interpret(meaningful_params, patch_np)

            patch_results.append(sum(complexity))

        image_results.append(torch.tensor(patch_results, dtype=torch.float64))

    return torch.stack(image_results) / (patch.shape[1] * patch.shape[2])  # normalize


class Meaningful(Complexity):
    def __init__(
        self,
        meaningful_params: dict,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        num_workers: int = 0,
    ) -> None:
        """
        meaningful_params: Parameters for the ComplexityMeasurer.
        """
        self.meaningful_params = meaningful_params
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_workers = num_workers

    def _compute(self, ds: ImageFolder) -> Any:
        return _compute_meaningful(
            ds=ds,
            meaningful_params=self.meaningful_params,
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
    elif config.startswith("variance"):
        _, neighborhood_size = config.split("_")

        return Variance(
            neighborhood_size=int(neighborhood_size),
            patch_size=patch_size,
            patch_stride=patch_stride,
            num_workers=num_workers,
        )
    elif config == "meaningful":
        meaningful_params = {
            "ncs_to_check": 12,
            "n_cluster_inits": 10,
            "nz": 4,
            "num_levels": 6,
            "cluster_model": "GMM",
            "info_subsample": 0.8,
            "suppress_all_prints": True
        }

        return Meaningful(
            meaningful_params=meaningful_params,
            patch_size=patch_size,
            patch_stride=patch_stride,
            num_workers=num_workers,
        )
    else:
        raise NotImplementedError(f"No matching complexity metric for {config}.")
