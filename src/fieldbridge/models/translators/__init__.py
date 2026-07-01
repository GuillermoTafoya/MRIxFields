"""Latent translator interfaces and implementations."""

from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.models.translators.identity import IdentityTranslator
from fieldbridge.models.translators.ot_cfm_stub import OTCFMTranslatorStub
from fieldbridge.models.translators.sb_stub import SchrodingerBridgeTranslatorStub

__all__ = [
    "BaseTranslator",
    "IdentityTranslator",
    "OTCFMTranslatorStub",
    "SchrodingerBridgeTranslatorStub",
]

