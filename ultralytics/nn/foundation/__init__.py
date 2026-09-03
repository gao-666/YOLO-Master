"""Opt-in, training-only Foundation Teacher interfaces."""

from .losses import cosine_kd_loss, foreground_token_weights, hybrid_kd_loss, relational_kd_loss
from .projectors import P4AlignmentProjector
from .protocol import FoundationFeatures, FoundationTeacher
from .response import (
    GLOBAL_BATCH_INDEX_VERSION,
    RESPONSE_FIELD_CONDITIONS,
    RESPONSE_FIELD_PAYLOAD_VERSION,
    BatchNormBufferSnapshot,
    ResponseFieldCondition,
    apply_response_field_condition_batch,
    build_response_field_paired_view,
    logical_global_batch_index,
    preserve_batchnorm_buffers,
    response_field_condition,
    response_field_kd_loss,
    response_field_noise_seed,
    strict_cosine_kd_loss,
    tensor_sha256,
)
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
    "GLOBAL_BATCH_INDEX_VERSION",
    "RESPONSE_FIELD_CONDITIONS",
    "RESPONSE_FIELD_PAYLOAD_VERSION",
    "BatchNormBufferSnapshot",
    "DINOv2Teacher",
    "DINOv3Teacher",
    "FoundationFeatures",
    "FoundationTeacher",
    "FoundationTeacherRouter",
    "MultiFoundationTeacher",
    "P4AlignmentProjector",
    "RegionSemanticProjector",
    "ResponseFieldCondition",
    "SigLIP2Teacher",
    "StudentFeatureTap",
    "apply_response_field_condition_batch",
    "build_response_field_paired_view",
    "cosine_kd_loss",
    "foreground_token_weights",
    "foundation_multiteacher_summary",
    "foundation_teacher_summary",
    "hybrid_kd_loss",
    "logical_global_batch_index",
    "positive_region_pool",
    "preserve_batchnorm_buffers",
    "region_image_loss",
    "region_text_loss",
    "relational_kd_loss",
    "response_field_condition",
    "response_field_kd_loss",
    "response_field_noise_seed",
    "routing_kd_loss",
    "semantic_distillation_loss",
    "strict_cosine_kd_loss",
    "tensor_sha256",
]
