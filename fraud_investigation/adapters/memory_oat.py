"""In-memory OAT / device / IP adapters (stand-ins for DB calls)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fraud_investigation.adapters.base import SORAdapter
from fraud_investigation.models.evidence import (
    AccountProfile,
    DeviceRecord,
    EvidenceBundle,
    InteractionNote,
    IpRecord,
    OatLoginEvent,
)
from fraud_investigation.models.schemas import Alert
from fraud_investigation.sample_data import store as sample


class MemoryDeviceAdapter(SORAdapter):
    name = "device_memory"

    def fetch(self, alert: Alert, context: dict[str, Any] | None = None) -> EvidenceBundle:
        raw = sample.DEVICES.get(alert.device_id)
        if not raw:
            return EvidenceBundle(source=self.name, payload={"device": None, "found": False})

        device = DeviceRecord(**raw)
        other_profiles = [
            AccountProfile(**sample.ACCOUNTS[aid])
            for aid in device.other_account_ids
            if aid in sample.ACCOUNTS
        ]
        payer = sample.ACCOUNTS.get(alert.customer_account_number)
        return EvidenceBundle(
            source=self.name,
            payload={
                "found": True,
                "device": device.model_dump(),
                "payer_profile": AccountProfile(**payer).model_dump() if payer else None,
                "other_profiles": [p.model_dump() for p in other_profiles],
                "as_of": datetime.now(timezone.utc).isoformat(),
            },
        )


class MemoryIpAdapter(SORAdapter):
    name = "ip_memory"

    def fetch(self, alert: Alert, context: dict[str, Any] | None = None) -> EvidenceBundle:
        context = context or {}
        ip_address = (
            context.get("ip_address")
            or (alert.metadata or {}).get("force_ip")
        )
        if not ip_address:
            # Prefer alerted/suspicious OAT event for this device
            events = [
                e
                for e in sample.OAT_EVENTS
                if e["customer_account_number"] == alert.customer_account_number
                and e["device_id"] == alert.device_id
                and (e.get("alerted") or e.get("suspicious"))
            ]
            if events:
                ip_address = events[-1]["ip_address"]

        raw = sample.IPS.get(ip_address) if ip_address else None
        payer = sample.ACCOUNTS.get(alert.customer_account_number)
        return EvidenceBundle(
            source=self.name,
            payload={
                "found": bool(raw),
                "ip": IpRecord(**raw).model_dump() if raw else None,
                "payer_profile": AccountProfile(**payer).model_dump() if payer else None,
                "company_isp_keywords": sample.COMPANY_ISP_KEYWORDS,
            },
        )


class MemoryOatAdapter(SORAdapter):
    name = "oat_memory"

    def fetch(self, alert: Alert, context: dict[str, Any] | None = None) -> EvidenceBundle:
        # Prefer same-device history; fall back to account-level 90-day window
        device_events = [
            e
            for e in sample.OAT_EVENTS
            if e["customer_account_number"] == alert.customer_account_number
            and e["device_id"] == alert.device_id
        ]
        account_events = [
            e
            for e in sample.OAT_EVENTS
            if e["customer_account_number"] == alert.customer_account_number
        ]
        chosen = device_events or account_events
        events = [OatLoginEvent(**e).model_dump() for e in chosen]
        return EvidenceBundle(
            source=self.name,
            payload={
                "lookback_days": 90,
                "events": events,
                "alert_device_id": alert.device_id,
            },
        )


class MemoryNotesAdapter(SORAdapter):
    name = "notes_memory"

    def fetch(self, alert: Alert, context: dict[str, Any] | None = None) -> EvidenceBundle:
        notes = [
            InteractionNote(**n).model_dump()
            for n in sample.INTERACTION_NOTES
            if n["customer_account_number"] == alert.customer_account_number
        ]
        return EvidenceBundle(
            source=self.name,
            payload={"notes": notes, "has_previous_interaction": len(notes) > 0},
        )


class HttpSORAdapter(SORAdapter):
    """Configurable HTTP adapter stub — wire real endpoints later via process YAML."""

    name = "http_generic"

    def __init__(self, base_url: str, path_template: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.path_template = path_template

    def fetch(self, alert: Alert, context: dict[str, Any] | None = None) -> EvidenceBundle:
        # Placeholder: do not call network in sample mode.
        path = self.path_template.format(
            alert_id=alert.alert_id,
            account=alert.customer_account_number,
            device_id=alert.device_id,
        )
        return EvidenceBundle(
            source=self.name,
            payload={
                "configured_url": f"{self.base_url}{path}",
                "status": "not_called_in_sample_mode",
                "hint": "Replace Memory* adapters with this adapter + real HTTP client.",
            },
        )
