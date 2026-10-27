"""IP research decision tree (deterministic + optional soft geo check)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fraud_investigation.models.schemas import FlagColor, RiskIndicator

IP_TENURE_DAYS = 90


def _as_naive_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def is_company_isp(isp: str | None, keywords: list[str]) -> bool:
    text = (isp or "").lower()
    return any(k.lower() in text for k in keywords)


def location_makes_sense(
    ip_location: str | None,
    payer_location: str | None,
    ip_city: str | None = None,
    payer_city: str | None = None,
) -> tuple[bool, str]:
    """Simple heuristic geo check.

    For richer interpretation (travel notes, impossible travel), plug an LLM
    interpreter behind the same function signature later.
    """
    ip_loc = (ip_location or ip_city or "").strip().lower()
    payer_loc = (payer_location or payer_city or "").strip().lower()

    if not ip_loc or not payer_loc:
        return False, "insufficient_location_data"

    # Match if payer city/location token appears in IP location (or vice versa)
    payer_tokens = {t for t in payer_loc.replace(",", " ").split() if len(t) > 2}
    ip_tokens = {t for t in ip_loc.replace(",", " ").split() if len(t) > 2}
    if payer_tokens & ip_tokens:
        return True, "location_overlap"

    return False, "location_mismatch"


def evaluate_ip(
    ip: dict[str, Any] | None,
    payer_profile: dict[str, Any] | None,
    company_isp_keywords: list[str] | None = None,
    as_of: datetime | None = None,
) -> list[RiskIndicator]:
    indicators: list[RiskIndicator] = []
    as_of = _as_naive_utc(as_of) or datetime.now(timezone.utc).replace(tzinfo=None)
    keywords = company_isp_keywords or []

    if not ip:
        indicators.append(
            RiskIndicator(
                code="IP_NOT_FOUND",
                source="oat.ip",
                flag=FlagColor.RED,
                message="IP record not found for alerted/suspicious session.",
                evidence={},
            )
        )
        return indicators

    isp = ip.get("isp")
    if is_company_isp(isp, keywords):
        indicators.append(
            RiskIndicator(
                code="IP_COMPANY_ISP",
                source="oat.ip",
                flag=FlagColor.GREEN,
                message=f"Company ISP detected ('{isp}'). No fraud per policy.",
                evidence={"isp": isp, "xa_id": ip.get("xa_id"), "location": ip.get("location")},
            )
        )
        return indicators

    first_seen = _as_naive_utc(ip.get("first_seen_at"))
    tenure_days = (as_of - first_seen).days if first_seen else 0

    if tenure_days > IP_TENURE_DAYS:
        indicators.append(
            RiskIndicator(
                code="IP_TENURE_OK",
                source="oat.ip",
                flag=FlagColor.GREEN,
                message=f"IP tenured {tenure_days} days (> {IP_TENURE_DAYS}). No fraud.",
                evidence={
                    "tenure_days": tenure_days,
                    "ip_address": ip.get("ip_address"),
                    "xa_id": ip.get("xa_id"),
                },
            )
        )
        return indicators

    payer_loc = (payer_profile or {}).get("location_label") or (payer_profile or {}).get("city")
    ok, reason = location_makes_sense(
        ip_location=ip.get("location"),
        payer_location=payer_loc,
        ip_city=ip.get("city"),
        payer_city=(payer_profile or {}).get("city"),
    )

    if ok:
        indicators.append(
            RiskIndicator(
                code="IP_LOCATION_OK",
                source="oat.ip",
                flag=FlagColor.GREEN,
                message="New IP but location makes sense vs customer profile. No risk.",
                evidence={
                    "reason": reason,
                    "ip_location": ip.get("location"),
                    "payer_location": payer_loc,
                    "xa_id": ip.get("xa_id"),
                    "isp": isp,
                },
            )
        )
    else:
        indicators.append(
            RiskIndicator(
                code="IP_LOCATION_RISK",
                source="oat.ip",
                flag=FlagColor.RED,
                message="New IP and location does not make sense vs customer profile.",
                evidence={
                    "reason": reason,
                    "ip_location": ip.get("location"),
                    "payer_location": payer_loc,
                    "xa_id": ip.get("xa_id"),
                    "isp": isp,
                    "tenure_days": tenure_days,
                },
            )
        )

    return indicators
