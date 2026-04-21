"""Plugin module discovery for AgenticSwarmBench.

Optional extension modules register themselves via ``pyproject.toml``
entry points in the ``asb.modules`` group.  The core codebase checks
for their presence at runtime and degrades gracefully when absent.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModuleInterface(Protocol):
    """Minimal contract every ASB plugin module must satisfy."""

    name: str

    def register_cli_flags(self, group: Any) -> None: ...
    def apply(self, context: dict) -> Any: ...


_MODULE_CACHE: dict[str, Any] | None = None


def discover_modules() -> dict[str, Any]:
    """Scan for installed ASB plugin modules via entry_points."""
    global _MODULE_CACHE
    if _MODULE_CACHE is not None:
        return _MODULE_CACHE

    modules: dict[str, Any] = {}
    try:
        eps = importlib.metadata.entry_points(group="asb.modules")
    except TypeError:
        eps = importlib.metadata.entry_points().get("asb.modules", [])

    for ep in eps:
        try:
            modules[ep.name] = ep.load()
        except Exception:
            pass

    _MODULE_CACHE = modules
    return modules


def get_module(name: str) -> Any | None:
    """Return a loaded module by name, or None if not installed."""
    return discover_modules().get(name)


def has_module(name: str) -> bool:
    """Check whether a module is installed without loading it."""
    return name in discover_modules()


def require_module(name: str, feature: str) -> Any:
    """Return a loaded module or raise a clear error explaining what to install."""
    module = get_module(name)
    if module is None:
        raise RuntimeError(
            f"{feature} requires the asb-{name.replace('_', '-')} module. "
            f"Install it with: pip install asb-{name.replace('_', '-')}"
        )
    return module
