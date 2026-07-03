"""Autoencoder interfaces and implementations."""

from fieldbridge.models.autoencoders.base import BaseDecoder, BaseEncoder
from fieldbridge.models.autoencoders.cnn_autoencoder import CNNDecoder, CNNEncoder
from fieldbridge.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder

__all__ = [
    "BaseDecoder",
    "BaseEncoder",
    "CNNDecoder",
    "CNNEncoder",
    "IdentityDecoder",
    "IdentityEncoder",
    "KLVAEDecoder",
    "KLVAEEncoder",
]

