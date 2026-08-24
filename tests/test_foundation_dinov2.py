"""Offline tests for the explicit DINOv2 Foundation Teacher contract."""

from types import SimpleNamespace

import torch
from torch import nn

from ultralytics.nn.foundation import DINOv2Teacher
from ultralytics.nn.foundation.preprocessing import DINOV2_IMAGE_MEAN, DINOV2_IMAGE_STD


class DummyDINOv2Backbone(nn.Module):
    """Minimal token-output backbone matching the Transformers DINOv2 interface."""

    def __init__(self, patch_size=4, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(patch_size=patch_size, hidden_size=hidden_size)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values):
        batch, _, height, width = pixel_values.shape
        patches = height // self.config.patch_size * (width // self.config.patch_size)
        tokens = self.scale * torch.ones(
            batch, 1 + patches, self.config.hidden_size, device=pixel_values.device, dtype=pixel_values.dtype
        )
        return SimpleNamespace(last_hidden_state=tokens)


def test_dinov2_contract_is_frozen_spatial_preserving_and_auditable():
    teacher = DINOv2Teacher(
        model_id="facebook/dinov2-small",
        revision="locked-revision",
        model=DummyDINOv2Backbone(),
        device="cpu",
    )
    features = teacher.encode(torch.zeros(2, 3, 5, 6))

    assert features.dense["p4"].shape == (2, 8, 2, 2)
    assert features.pooled.shape == (2, 8)
    assert features.metadata["teacher_family"] == "dinov2"
    assert features.metadata["revision"] == "locked-revision"
    assert features.metadata["input_size"] == (5, 6)
    assert features.metadata["padded_size"] == (8, 8)
    assert features.metadata["preprocessing"] == {
        "contract": "dinov2_dense_spatial_preserving_v1",
        "input_range": [0.0, 1.0],
        "resize": False,
        "center_crop": False,
        "padding": "bottom_right_to_patch_multiple",
        "mean": list(DINOV2_IMAGE_MEAN),
        "std": list(DINOV2_IMAGE_STD),
    }
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_dinov2_preprocessing_uses_family_specific_constants():
    teacher = DINOv2Teacher(model=DummyDINOv2Backbone(patch_size=1), device="cpu")
    result = teacher.preprocess(torch.zeros(1, 3, 1, 1)).flatten()
    expected = torch.tensor([-mean / std for mean, std in zip(DINOV2_IMAGE_MEAN, DINOV2_IMAGE_STD)])
    assert torch.allclose(result, expected)
