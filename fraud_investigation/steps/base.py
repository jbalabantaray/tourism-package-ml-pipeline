"""Investigation step plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fraud_investigation.adapters.base import AdapterRegistry
from fraud_investigation.models.schemas import Alert, InvestigationStepResult


class InvestigationStep(ABC):
    step_id: str = "base"

    @abstractmethod
    def run(
        self,
        alert: Alert,
        adapters: AdapterRegistry,
        context: dict[str, Any],
    ) -> InvestigationStepResult:
        raise NotImplementedError


class StepRegistry:
    def __init__(self) -> None:
        self._steps: dict[str, InvestigationStep] = {}

    def register(self, key: str, step: InvestigationStep) -> None:
        self._steps[key] = step

    def get(self, key: str) -> InvestigationStep:
        if key not in self._steps:
            raise KeyError(f"Investigation step '{key}' is not registered")
        return self._steps[key]
