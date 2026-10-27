"""Deterministic relationship checks between payer and other device users."""

from __future__ import annotations

from typing import Any


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def accounts_related(payer: dict[str, Any], other: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (related?, matched_signals).

    Related if they share same address, same last name, same location,
    or have a joint account relationship.
    """
    reasons: list[str] = []

    if payer.get("account_number") in (other.get("joint_account_with") or []) or other.get(
        "account_number"
    ) in (payer.get("joint_account_with") or []):
        reasons.append("joint_account")

    if _norm(payer.get("last_name")) and _norm(payer.get("last_name")) == _norm(
        other.get("last_name")
    ):
        reasons.append("same_last_name")

    payer_addr = " ".join(
        [
            _norm(payer.get("address_line")),
            _norm(payer.get("city")),
            _norm(payer.get("state")),
            _norm(payer.get("postal_code")),
        ]
    ).strip()
    other_addr = " ".join(
        [
            _norm(other.get("address_line")),
            _norm(other.get("city")),
            _norm(other.get("state")),
            _norm(other.get("postal_code")),
        ]
    ).strip()
    if payer_addr and payer_addr == other_addr:
        reasons.append("same_address")

    if _norm(payer.get("location_label")) and _norm(payer.get("location_label")) == _norm(
        other.get("location_label")
    ):
        reasons.append("same_location")

    # De-dupe while preserving order
    seen: set[str] = set()
    unique = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return (len(unique) > 0, unique)
