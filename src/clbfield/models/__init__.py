"""Model contracts and baseline implementations."""

from clbfield.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from clbfield.models.conditioning import DomainConditioner
from clbfield.models.factory import build_decoder, build_encoder, build_translator
from clbfield.models.film import FiLMLayer
from clbfield.models.translators.identity import IdentityTranslator

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

