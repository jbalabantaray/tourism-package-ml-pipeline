from __future__ import annotations

from typing import Any

from fraud_investigation.adapters.base import AdapterRegistry
from fraud_investigation.models.schemas import (
    Alert,
    DispositionType,
    FlagColor,
    InvestigationStepResult,
    RiskIndicator,
)
from fraud_investigation.steps.base import InvestigationStep


class AggregateStep(InvestigationStep):
    """Roll up risk indicators into contact vs disposition recommendation."""

    step_id = "aggregate"

    def run(
        self,
        alert: Alert,
        adapters: AdapterRegistry,
        context: dict[str, Any],
    ) -> InvestigationStepResult:
        all_indicators: list[RiskIndicator] = context.get("all_indicators") or []
        red = [i for i in all_indicators if i.flag == FlagColor.RED]
        amber = [i for i in all_indicators if i.flag == FlagColor.AMBER]

        contact = False
        disposition = DispositionType.NO_FRAUD
        recommendation = "No risk indicators present. Disposition: No fraud."

        if red:
            contact = True
            disposition = DispositionType.PENDING
            recommendation = (
                "Risk indicator(s) present. Contact the customer before final disposition."
            )
        elif amber:
            contact = alert.auto_contact_eligible
            disposition = DispositionType.PENDING
            recommendation = (
                "Ambiguous indicators present. Keep Pending"
                + (" and auto-contact customer." if contact else "; investigator review required.")
            )

        # Auto-contact eligible path can force contact even without reds if configured later
        if alert.auto_contact_eligible and red:
            contact = True

        context["disposition"] = disposition
        context["contact_customer"] = contact
        context["recommendation"] = recommendation

        return InvestigationStepResult(
            step_id=self.step_id,
            indicators=[],
            notes=[recommendation],
            raw={
                "red_count": len(red),
                "amber_count": len(amber),
                "disposition": disposition.value,
                "contact_customer": contact,
            },
        )
