from __future__ import annotations

from typing import Any

from fraud_investigation.adapters.base import AdapterRegistry
from fraud_investigation.models.schemas import (
    Alert,
    FlagColor,
    InvestigationStepResult,
    RiskIndicator,
)
from fraud_investigation.steps.base import InvestigationStep


class PreviousInteractionStep(InvestigationStep):
    step_id = "previous_interaction"

    def run(
        self,
        alert: Alert,
        adapters: AdapterRegistry,
        context: dict[str, Any],
    ) -> InvestigationStepResult:
        evidence = adapters.get("notes").fetch(alert, context)
        notes = evidence.payload.get("notes") or []
        has_prev = bool(evidence.payload.get("has_previous_interaction"))

        indicators: list[RiskIndicator] = []
        notes_out: list[str] = []

        if has_prev:
            verified = [n for n in notes if n.get("verified_activity")]
            if verified:
                indicators.append(
                    RiskIndicator(
                        code="PREV_INTERACTION_VERIFIED",
                        source="notes",
                        flag=FlagColor.GREEN,
                        message="Previous customer interaction found; activity verified from notes.",
                        evidence={"note_ids": [n["note_id"] for n in verified]},
                    )
                )
                notes_out.append("Verified prior interaction notes present.")
            else:
                # Free-text notes without verified flag — LLM can interpret later
                indicators.append(
                    RiskIndicator(
                        code="PREV_INTERACTION_NEEDS_REVIEW",
                        source="notes",
                        flag=FlagColor.AMBER,
                        message="Previous interaction exists but not auto-verified; review notes.",
                        evidence={"note_ids": [n["note_id"] for n in notes]},
                        deterministic=False,
                    )
                )
                notes_out.append("Prior notes need manual/LLM verification.")
            context["skip_deep_research"] = False  # still continue deep research in sample
            context["has_previous_interaction"] = True
        else:
            notes_out.append("No previous interaction; proceed to OAT research.")
            context["has_previous_interaction"] = False

        return InvestigationStepResult(
            step_id=self.step_id,
            indicators=indicators,
            notes=notes_out,
            raw=evidence.payload,
        )
