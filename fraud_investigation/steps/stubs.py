"""Pluggable stubs for Transaction/Payee and Customer Profile reviews.

Wire real SOR adapters + rules via process YAML when maps are ready.
"""

from __future__ import annotations

from typing import Any

from fraud_investigation.adapters.base import AdapterRegistry
from fraud_investigation.models.schemas import Alert, InvestigationStepResult
from fraud_investigation.steps.base import InvestigationStep


class TransactionPayeeReviewStep(InvestigationStep):
    step_id = "transaction_payee_review"

    def run(
        self,
        alert: Alert,
        adapters: AdapterRegistry,
        context: dict[str, Any],
    ) -> InvestigationStepResult:
        return InvestigationStepResult(
            step_id=self.step_id,
            status="skipped_stub",
            notes=[
                "Stub: plug payee relationship/history/ownership SOR + rules here.",
            ],
            raw={},
        )


class CustomerProfileReviewStep(InvestigationStep):
    step_id = "customer_profile_review"

    def run(
        self,
        alert: Alert,
        adapters: AdapterRegistry,
        context: dict[str, Any],
    ) -> InvestigationStepResult:
        return InvestigationStepResult(
            step_id=self.step_id,
            status="skipped_stub",
            notes=[
                "Stub: plug OTP history, contact info, hard holds SOR + rules here.",
            ],
            raw={},
        )
