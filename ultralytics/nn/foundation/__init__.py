"""Opt-in, training-only Foundation Teacher interfaces."""

from .losses import cosine_kd_loss, foreground_token_weights, hybrid_kd_loss, relational_kd_loss
from .projectors import P4AlignmentProjector
from .protocol import FoundationFeatures, FoundationTeacher
from .routing import (
    FoundationTeacherRouter,
    foundation_multiteacher_summary,
    foundation_teacher_summary,
    routing_kd_loss,
)
from .semantic import (
    RegionSemanticProjector,
    positive_region_pool,
    region_image_loss,
    region_text_loss,
    semantic_distillation_loss,
)
from .taps import StudentFeatureTap
from .teachers import (
    DEFAULT_DINOV2_MODEL,
    DEFAULT_DINOV3_MODEL,
    DEFAULT_SIGLIP2_MODEL,
    DINOv2Teacher,
    DINOv3Teacher,
    MultiFoundationTeacher,
    SigLIP2Teacher,
)

__all__ = [
    "DEFAULT_DINOV2_MODEL",
    "DEFAULT_DINOV3_MODEL",
    "DEFAULT_SIGLIP2_MODEL",
    "DINOv2Teacher",
    "DINOv3Teacher",
    "FoundationFeatures",
    "FoundationTeacher",
    "FoundationTeacherRouter",
    "MultiFoundationTeacher",
    "P4AlignmentProjector",
    "RegionSemanticProjector",
    "SigLIP2Teacher",
    "StudentFeatureTap",
    "cosine_kd_loss",
    "foreground_token_weights",
    "foundation_multiteacher_summary",
    "foundation_teacher_summary",
    "hybrid_kd_loss",
    "positive_region_pool",
    "region_image_loss",
    "region_text_loss",
    "relational_kd_loss",
    "routing_kd_loss",
    "semantic_distillation_loss",
]
