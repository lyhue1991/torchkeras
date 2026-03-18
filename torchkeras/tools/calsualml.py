"""
Compatibility shim for the optional ``causalml`` dependency.

Importing :mod:`torchkeras.tools.calsualml` lazily loads ``causalml`` and
provides a clearer error message when the library is missing. This keeps
``torchkeras.tools`` importable even when ``causalml`` is not installed.
"""

from functools import lru_cache
from importlib import import_module
from types import ModuleType

_MISSING_ERROR = (
    "`causalml` is required for torchkeras.tools.calsualml. "
    "Install it with `pip install causalml`."
)


@lru_cache()
def get_causalml() -> ModuleType:
    """
    Load and return the :mod:`causalml` module.

    Raises:
        ModuleNotFoundError: if ``causalml`` is not installed.
    """
    try:
        return import_module("causalml")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING_ERROR) from exc


def __getattr__(name: str):
    """Proxy attribute access to the underlying ``causalml`` module."""
    causalml = get_causalml()
    return getattr(causalml, name)


def version() -> str:
    """Return the installed ``causalml`` version (empty string if unknown)."""
    causalml = get_causalml()
    return getattr(causalml, "__version__", "")


__all__ = ["get_causalml", "version"]
