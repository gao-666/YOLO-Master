"""Foundation Teacher backend implementations."""

from .dinov2 import DEFAULT_DINOV2_MODEL, DINOv2Teacher
from .dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Teacher
from .multi import MultiFoundationTeacher
from .siglip2 import DEFAULT_SIGLIP2_MODEL, SigLIP2Teacher

__all__ = [
    "DEFAULT_DINOV2_MODEL",
    "DEFAULT_DINOV3_MODEL",
    "DEFAULT_SIGLIP2_MODEL",
    "DINOv2Teacher",
    "DINOv3Teacher",
    "MultiFoundationTeacher",
    "SigLIP2Teacher",
]
