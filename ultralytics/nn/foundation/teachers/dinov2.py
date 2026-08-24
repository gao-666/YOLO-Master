"""DINOv2 Foundation Teacher backend with an explicit dense-feature preprocessing contract."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn

from ..preprocessing import DINOV2_IMAGE_MEAN, DINOV2_IMAGE_STD, prepare_image_tensor
from ..protocol import FoundationFeatures
from .dinov3 import DINOv3Teacher, _is_auto

DEFAULT_DINOV2_MODEL = "facebook/dinov2-small"


class DINOv2Teacher(DINOv3Teacher):
    """Frozen DINOv2 encoder implementing the common Foundation Teacher protocol.

    The public Hugging Face image processor resizes and center-crops classification inputs. Dense P4 distillation
    instead preserves the YOLO input geometry, pads only to the patch multiple, and applies the DINOv2 normalization
    constants. This deliberate spatial-preserving deviation is recorded in every feature metadata payload.
    """

    name = "dinov2"

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV2_MODEL,
        *,
        revision: str | None = None,
        dtype: str | torch.dtype = "auto",
        device: str | int | torch.device = "auto",
        weights_path: str | Path | None = None,
        model: nn.Module | None = None,
        model_loader: Callable[..., nn.Module] | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.revision = str(revision) if revision is not None else None
        super().__init__(
            model_id=model_id,
            dtype=dtype,
            device=device,
            weights_path=weights_path,
            model=model,
            model_loader=model_loader,
            local_files_only=local_files_only,
        )

    def _load_model(self, model_loader: Callable[..., nn.Module] | None) -> nn.Module:
        """Load DINOv2 lazily while retaining the inherited offline injection contract."""
        if model_loader is not None:
            return super()._load_model(model_loader)
        try:
            from transformers import Dinov2Model
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Foundation DINOv2 backend requires optional dependency 'transformers>=4.56.0,<6'. "
                "Install with: pip install -e '.[foundation]'"
            ) from exc

        local_directory = bool(self.weights_path and Path(self.weights_path).is_dir())
        source = self.weights_path if local_directory else self.model_id
        kwargs = {"local_files_only": self.local_files_only}
        if self.revision is not None and not local_directory:
            kwargs["revision"] = self.revision
        if not _is_auto(self._dtype_request):
            kwargs["torch_dtype"] = self._resolve_dtype(self._dtype_request)
        model = Dinov2Model.from_pretrained(source, **kwargs)
        if self.weights_path and Path(self.weights_path).is_file():
            self._load_state_dict(model, Path(self.weights_path))
        return model

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the declared spatial-preserving DINOv2 dense-feature contract."""
        return prepare_image_tensor(
            images.to(device=self.device),
            patch_size=self.patch_size,
            mean=DINOV2_IMAGE_MEAN,
            std=DINOV2_IMAGE_STD,
        ).to(dtype=self.dtype)

    def encode(self, images: torch.Tensor) -> FoundationFeatures:
        """Encode images and attach the exact DINOv2 preprocessing identity used for distillation."""
        features = super().encode(images)
        features.metadata.update(
            {
                "teacher_family": self.name,
                "revision": self.revision,
                "preprocessing": {
                    "contract": "dinov2_dense_spatial_preserving_v1",
                    "input_range": [0.0, 1.0],
                    "resize": False,
                    "center_crop": False,
                    "padding": "bottom_right_to_patch_multiple",
                    "mean": list(DINOV2_IMAGE_MEAN),
                    "std": list(DINOV2_IMAGE_STD),
                },
            }
        )
        return features


__all__ = ["DEFAULT_DINOV2_MODEL", "DINOv2Teacher"]
