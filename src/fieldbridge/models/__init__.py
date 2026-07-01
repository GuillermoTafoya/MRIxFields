"""Model contracts and baseline implementations."""

from fieldbridge.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from fieldbridge.models.conditioning import DomainConditioner
from fieldbridge.models.factory import build_decoder, build_encoder, build_translator
from fieldbridge.models.film import FiLMLayer
from fieldbridge.models.translators.identity import IdentityTranslator

__all__ = [
    "DomainConditioner",
    "FiLMLayer",
    "IdentityDecoder",
    "IdentityEncoder",
    "IdentityTranslator",
    "build_decoder",
    "build_encoder",
    "build_translator",
]

