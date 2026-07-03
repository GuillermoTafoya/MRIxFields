"""Model contracts and baseline implementations."""

from fieldbridge.models.autoencoders.cnn_autoencoder import CNNDecoder, CNNEncoder
from fieldbridge.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from fieldbridge.models.conditioning import DomainConditioner, DomainEmbedding
from fieldbridge.models.factory import build_decoder, build_encoder, build_translator
from fieldbridge.models.film import FiLMGroupNorm, FiLMLayer
from fieldbridge.models.translators.conditional_cnn import ConditionalCNNFieldTranslator
from fieldbridge.models.translators.identity import IdentityTranslator

__all__ = [
    "DomainConditioner",
    "DomainEmbedding",
    "FiLMGroupNorm",
    "FiLMLayer",
    "CNNDecoder",
    "CNNEncoder",
    "ConditionalCNNFieldTranslator",
    "IdentityDecoder",
    "IdentityEncoder",
    "IdentityTranslator",
    "build_decoder",
    "build_encoder",
    "build_translator",
]

