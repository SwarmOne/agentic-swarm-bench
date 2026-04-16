"""Shared utilities for proxy and recorder modules."""

from __future__ import annotations

_ANTHROPIC_HOSTS = ("api.anthropic.com", "anthropic.com")


def _detect_upstream_api(upstream_url: str, explicit: str | None) -> str:
    """Return 'anthropic' or 'openai' based on explicit flag or URL heuristic."""
    if explicit:
        return explicit
    from urllib.parse import urlparse

    host = urlparse(upstream_url).hostname or ""
    if any(host.endswith(h) for h in _ANTHROPIC_HOSTS):
        return "anthropic"
    return "openai"
