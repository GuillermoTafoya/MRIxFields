"""Latent translator interfaces and implementations."""

from clbfield.models.translators.base import BaseTranslator
from clbfield.models.translators.identity import IdentityTranslator
from clbfield.models.translators.ot_cfm_stub import OTCFMTranslatorStub
from clbfield.models.translators.sb_stub import SchrodingerBridgeTranslatorStub

__all__ = [
    "BaseTranslator",
    "IdentityTranslator",
    "OTCFMTranslatorStub",
    "SchrodingerBridgeTranslatorStub",
]

