"""Device research decision tree (deterministic)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fraud_investigation.models.schemas import FlagColor, RiskIndicator
from fraud_investigation.rules.relationship import accounts_related

DEVICE_TENURE_DAYS = 90


def _as_naive_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def evaluate_device(
    device: dict[str, Any] | None,
    payer_profile: dict[str, Any] | None,
    other_profiles: list[dict[str, Any]],
    as_of: datetime | None = None,
) -> list[RiskIndicator]:
    indicators: list[RiskIndicator] = []
    as_of = _as_naive_utc(as_of) or datetime.now(timezone.utc).replace(tzinfo=None)

    if not device:
        indicators.append(
            RiskIndicator(
                code="DEVICE_NOT_FOUND",
                source="oat.device",
                flag=FlagColor.RED,
                message="Device id from alert was not found in device SOR.",
                evidence={},
            )
        )
        return indicators

    first_seen = _as_naive_utc(device.get("first_seen_at"))
    tenure_days = (as_of - first_seen).days if first_seen else 0

    if tenure_days > DEVICE_TENURE_DAYS:
        indicators.append(
            RiskIndicator(
                code="DEVICE_TENURE_OK",
                source="oat.device",
                flag=FlagColor.GREEN,
                message=f"Device tenured {tenure_days} days (> {DEVICE_TENURE_DAYS}). No risk.",
                evidence={"tenure_days": tenure_days, "device_id": device["device_id"]},
            )
        )
        return indicators

    other_ids = device.get("other_account_ids") or []
    if not other_ids:
        indicators.append(
            RiskIndicator(
                code="DEVICE_SOLE_USER_NEW",
                source="oat.device",
                flag=FlagColor.GREEN,
                message=(
                    f"Device is new ({tenure_days} days) but no other users on device. No risk."
                ),
                evidence={"tenure_days": tenure_days, "device_id": device["device_id"]},
            )
        )
        return indicators

    # Other users present — check relationship to payer
    if not payer_profile:
        indicators.append(
            RiskIndicator(
                code="DEVICE_OTHER_USERS_PAYER_UNKNOWN",
                source="oat.device",
                flag=FlagColor.AMBER,
                message="Other users on new device but payer profile missing; needs review.",
                evidence={"other_account_ids": other_ids},
            )
        )
        return indicators

    unrelated: list[dict[str, Any]] = []
    related: list[dict[str, Any]] = []
    for other in other_profiles:
        is_related, reasons = accounts_related(payer_profile, other)
        entry = {"account": other.get("account_number"), "reasons": reasons}
        if is_related:
            related.append(entry)
        else:
            unrelated.append(entry)

    # Also treat unknown other accounts (no profile) as unrelated
    known = {p.get("account_number") for p in other_profiles}
    for aid in other_ids:
        if aid not in known:
            unrelated.append({"account": aid, "reasons": []})

    if unrelated:
        indicators.append(
            RiskIndicator(
                code="DEVICE_UNRELATED_COUSER",
                source="oat.device",
                flag=FlagColor.RED,
                message=(
                    "New device has other user(s) with no relationship to payer "
                    "(address/last name/location/joint account)."
                ),
                evidence={
                    "tenure_days": tenure_days,
                    "unrelated": unrelated,
                    "related": related,
                },
            )
        )
    else:
        indicators.append(
            RiskIndicator(
                code="DEVICE_RELATED_COUSER",
                source="oat.device",
                flag=FlagColor.GREEN,
                message="New device has other users, but all are related to payer. No risk.",
                evidence={"tenure_days": tenure_days, "related": related},
            )
        )

    return indicators
