from __future__ import annotations

from typing import Any

from fraud_investigation.adapters.base import AdapterRegistry
from fraud_investigation.models.schemas import (
    Alert,
    AlertType,
    DispositionType,
    FlagColor,
    InvestigationResult,
    InvestigationStepResult,
    RiskIndicator,
)
from fraud_investigation.orchestrator.loader import load_process
from fraud_investigation.orchestrator.registry import build_adapter_registry, build_step_registry
from fraud_investigation.steps.base import StepRegistry


class InvestigationOrchestrator:
    """Config-driven investigation runner.

    Process YAML selects adapter keys and step order. Steps remain unaware of
    whether SOR data came from DB or HTTP.
    """

    def __init__(
        self,
        adapter_registry: AdapterRegistry | None = None,
        step_registry: StepRegistry | None = None,
    ) -> None:
        self.adapter_registry = adapter_registry or build_adapter_registry()
        self.step_registry = step_registry or build_step_registry()

    def resolve_process_id(self, alert: Alert) -> str:
        # Extend with alert-type → process map as more flows are added
        if alert.alert_type == AlertType.ONLINE_FRAUD:
            return "online_fraud_v1"
        return "online_fraud_v1"

    def run(self, alert: Alert, process_id: str | None = None) -> InvestigationResult:
        process_id = process_id or self.resolve_process_id(alert)
        process = load_process(process_id)

        # Bind logical adapter names from YAML to concrete adapters
        bound = AdapterRegistry()
        for logical_name, concrete_key in (process.get("adapters") or {}).items():
            bound.register(logical_name, self.adapter_registry.get(concrete_key))

        context: dict[str, Any] = {"all_indicators": []}
        step_results: list[InvestigationStepResult] = []

        for step_def in process.get("steps") or []:
            step = self.step_registry.get(step_def["type"])
            result = step.run(alert, bound, context)
            step_results.append(result)
            context["all_indicators"].extend(result.indicators)

        all_indicators: list[RiskIndicator] = context["all_indicators"]
        reds = [i for i in all_indicators if i.flag == FlagColor.RED]
        greens = [i for i in all_indicators if i.flag == FlagColor.GREEN]

        disposition = context.get("disposition") or (
            DispositionType.PENDING if reds else DispositionType.NO_FRAUD
        )
        contact = bool(context.get("contact_customer"))
        recommendation = context.get("recommendation") or (
            "Risk indicators present. Contact customer."
            if reds
            else "No risk indicators. Disposition: No fraud."
        )

        return InvestigationResult(
            alert_id=alert.alert_id,
            alert_type=alert.alert_type,
            disposition=disposition,
            recommendation=recommendation,
            contact_customer=contact,
            risk_indicators=[i for i in all_indicators if i.flag != FlagColor.GREEN],
            green_flags=greens,
            step_results=step_results,
            process_id=process.get("process_id", process_id),
            process_version=str(process.get("version", "0")),
        )
