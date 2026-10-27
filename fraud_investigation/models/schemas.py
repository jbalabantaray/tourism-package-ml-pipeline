"""Core domain schemas for fraud investigation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DispositionType(str, Enum):
    FRAUD = "Fraud"
    NO_FRAUD = "No fraud"
    PENDING = "Pending"


class FlagColor(str, Enum):
    RED = "RED"
    GREEN = "GREEN"
    AMBER = "AMBER"


class AlertType(str, Enum):
    ONLINE_FRAUD = "ONLINE_FRAUD"
    # Extend as new alert families are onboarded
    UNKNOWN = "UNKNOWN"


class Alert(BaseModel):
    alert_id: str
    alert_type: AlertType
    customer_account_number: str
    device_id: str
    auto_contact_eligible: bool = False
    created_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskIndicator(BaseModel):
    code: str
    source: str  # e.g. oat.device, oat.ip, notes
    flag: FlagColor
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    deterministic: bool = True


class InvestigationStepResult(BaseModel):
    step_id: str
    status: str = "completed"
    indicators: list[RiskIndicator] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class InvestigationResult(BaseModel):
    alert_id: str
    alert_type: AlertType
    disposition: DispositionType
    recommendation: str
    contact_customer: bool
    risk_indicators: list[RiskIndicator]
    green_flags: list[RiskIndicator]
    step_results: list[InvestigationStepResult]
    process_id: str
    process_version: str


class InvestigateRequest(BaseModel):
    alert: Alert
    # Optional overrides for local/demo runs
    dry_run: bool = False


class InvestigateResponse(BaseModel):
    result: InvestigationResult
