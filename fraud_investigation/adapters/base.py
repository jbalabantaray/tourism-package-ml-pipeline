"""SOR adapter interface — swap DB for API without changing steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fraud_investigation.models.evidence import EvidenceBundle
from fraud_investigation.models.schemas import Alert


class SORAdapter(ABC):
    """Fetch system-of-record data for an investigation step."""

    name: str = "base"

    @abstractmethod
    def fetch(self, alert: Alert, context: dict[str, Any] | None = None) -> EvidenceBundle:
        raise NotImplementedError


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SORAdapter] = {}

    def register(self, key: str, adapter: SORAdapter) -> None:
        self._adapters[key] = adapter

    def get(self, key: str) -> SORAdapter:
        if key not in self._adapters:
            raise KeyError(f"SOR adapter '{key}' is not registered")
        return self._adapters[key]

    def keys(self) -> list[str]:
        return list(self._adapters.keys())
