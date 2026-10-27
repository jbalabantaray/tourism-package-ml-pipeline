"""Evidence bundles returned by SOR adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeviceRecord(BaseModel):
    device_id: str
    customer_account_number: str
    first_seen_at: datetime
    last_seen_at: Optional[datetime] = None
    other_account_ids: list[str] = Field(default_factory=list)


class AccountProfile(BaseModel):
    account_number: str
    last_name: Optional[str] = None
    address_line: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    joint_account_with: list[str] = Field(default_factory=list)
    location_label: Optional[str] = None


class IpRecord(BaseModel):
    ip_address: str
    isp: Optional[str] = None
    xa_id: Optional[str] = None  # session / XA correlation id
    location: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    customer_account_number: Optional[str] = None


class OatLoginEvent(BaseModel):
    event_id: str
    customer_account_number: str
    device_id: str
    ip_address: str
    login_at: datetime
    suspicious: bool = False
    alerted: bool = False
    xa_id: Optional[str] = None
    location: Optional[str] = None
    isp: Optional[str] = None


class InteractionNote(BaseModel):
    note_id: str
    customer_account_number: str
    created_at: datetime
    channel: str
    text: str
    verified_activity: bool = False


class EvidenceBundle(BaseModel):
    """Generic bag of SOR payloads for a step."""

    source: str
    fetched_at: datetime = Field(default_factory=utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)
