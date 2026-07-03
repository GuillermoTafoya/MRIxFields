"""Conditional latent diffuser for Etapa 1 (field-strength conditioning, FiLM-based)."""

from fieldbridge.models.diffusion.denoising_unet import ConditionedResidualBlock, DenoisingUNet
from fieldbridge.models.diffusion.field_conditioner import FieldStrengthConditioner
from fieldbridge.models.diffusion.schedule import DiffusionSchedule, make_schedule, q_sample
from fieldbridge.models.diffusion.timestep_embedding import sinusoidal_timestep_embedding

__all__ = [
    "ConditionedResidualBlock",
    "DenoisingUNet",
    "DiffusionSchedule",
    "FieldStrengthConditioner",
    "make_schedule",
    "q_sample",
    "sinusoidal_timestep_embedding",
]
