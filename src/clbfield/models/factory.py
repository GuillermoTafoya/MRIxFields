"""Name-based construction for encoders, decoders, and translators.

Extended by each ladder stage (e.g. Fase C adds "stargan_v2_latent", Fase D adds
"ot_cfm") so the CLI and training loop don't need to import every concrete model
class by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from clbfield.models.autoencoders.base import BaseDecoder, BaseEncoder
from clbfield.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from clbfield.models.translators.base import BaseTranslator
from clbfield.models.translators.identity import IdentityTranslator

T = TypeVar("T")

_ENCODERS: dict[str, Callable[..., BaseEncoder]] = {"identity": IdentityEncoder}
_DECODERS: dict[str, Callable[..., BaseDecoder]] = {"identity": IdentityDecoder}
_TRANSLATORS: dict[str, Callable[..., BaseTranslator]] = {"identity": IdentityTranslator}


def build_encoder(name: str, **kwargs: Any) -> BaseEncoder:
    return _build(_ENCODERS, "encoder", name, **kwargs)


def build_decoder(name: str, **kwargs: Any) -> BaseDecoder:
    return _build(_DECODERS, "decoder", name, **kwargs)


def build_translator(name: str, **kwargs: Any) -> BaseTranslator:
    return _build(_TRANSLATORS, "translator", name, **kwargs)


def _build(registry: dict[str, Callable[..., T]], kind: str, name: str, **kwargs: Any) -> T:
    try:
        constructor = registry[name]
    except KeyError as exc:
        raise ValueError(f"Unknown {kind} {name!r}. Available: {sorted(registry)}.") from exc
    return constructor(**kwargs)
