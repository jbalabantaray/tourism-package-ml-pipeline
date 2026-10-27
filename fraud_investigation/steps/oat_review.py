from __future__ import annotations

from typing import Any

from fraud_investigation.adapters.base import AdapterRegistry
from fraud_investigation.models.schemas import (
    Alert,
    InvestigationStepResult,
)
from fraud_investigation.rules.device_rules import evaluate_device
from fraud_investigation.rules.ip_rules import evaluate_ip
from fraud_investigation.steps.base import InvestigationStep


class OatReviewStep(InvestigationStep):
    """OAT review: 90-day activity + device research + IP research."""

    step_id = "oat_review"

    def run(
        self,
        alert: Alert,
        adapters: AdapterRegistry,
        context: dict[str, Any],
    ) -> InvestigationStepResult:
        oat = adapters.get("oat").fetch(alert, context)
        device_ev = adapters.get("device").fetch(alert, context)

        # Prefer IP from alerted/suspicious OAT event
        events = oat.payload.get("events") or []
        focus = next((e for e in reversed(events) if e.get("alerted") or e.get("suspicious")), None)
        ip_ctx = {"ip_address": focus["ip_address"]} if focus else {}
        ip_ev = adapters.get("ip").fetch(alert, {**context, **ip_ctx})

        indicators: list[RiskIndicator] = []
        notes: list[str] = []

        # Summarize OAT window
        suspicious = [e for e in events if e.get("suspicious") or e.get("alerted")]
        notes.append(
            f"Reviewed {len(events)} OAT event(s) in lookback; "
            f"{len(suspicious)} suspicious/alerted."
        )
        # Alerted/suspicious login is why we are here — record as evidence note only,
        # not as a disposition-driving risk indicator.
        if suspicious:
            notes.append(
                "Focus events: "
                + ", ".join(
                    f"{e['event_id']}(xa={e.get('xa_id')})" for e in suspicious
                )
            )

        # Device tree
        device_indicators = evaluate_device(
            device=device_ev.payload.get("device"),
            payer_profile=device_ev.payload.get("payer_profile"),
            other_profiles=device_ev.payload.get("other_profiles") or [],
        )
        indicators.extend(device_indicators)

        # IP tree
        ip_indicators = evaluate_ip(
            ip=ip_ev.payload.get("ip"),
            payer_profile=ip_ev.payload.get("payer_profile"),
            company_isp_keywords=ip_ev.payload.get("company_isp_keywords") or [],
        )
        indicators.extend(ip_indicators)

        context["oat_events"] = events
        context["device_evidence"] = device_ev.payload
        context["ip_evidence"] = ip_ev.payload

        return InvestigationStepResult(
            step_id=self.step_id,
            indicators=indicators,
            notes=notes,
            raw={
                "oat": oat.payload,
                "device": device_ev.payload,
                "ip": ip_ev.payload,
            },
        )
