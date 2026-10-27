"""FastAPI entrypoint for fraud investigation."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from fraud_investigation.models.schemas import (
    Alert,
    AlertType,
    InvestigateRequest,
    InvestigateResponse,
)
from fraud_investigation.orchestrator.engine import InvestigationOrchestrator
from fraud_investigation.orchestrator.loader import list_processes

app = FastAPI(
    title="Fraud Investigation API",
    description=(
        "Config-driven fraud investigation orchestrator. "
        "Deterministic rules for device/IP; pluggable SORs; LLM hook for unstructured data."
    ),
    version="0.1.0",
)

orchestrator = InvestigationOrchestrator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/processes")
def processes() -> dict[str, list[str]]:
    return {"processes": list_processes()}


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest) -> InvestigateResponse:
    try:
        result = orchestrator.run(req.alert)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return InvestigateResponse(result=result)


@app.post("/investigate/demo/{scenario}", response_model=InvestigateResponse)
def investigate_demo(scenario: str) -> InvestigateResponse:
    """Run canned scenarios against sample SOR data."""
    demos: dict[str, Alert] = {
        "trusted_device": Alert(
            alert_id="ALT-DEMO-1",
            alert_type=AlertType.ONLINE_FRAUD,
            customer_account_number="ACC-1001",
            device_id="DEV-TRUSTED-001",
            auto_contact_eligible=False,
        ),
        "new_unrelated_device": Alert(
            alert_id="ALT-DEMO-2",
            alert_type=AlertType.ONLINE_FRAUD,
            customer_account_number="ACC-1001",
            device_id="DEV-NEW-UNRELATED",
            auto_contact_eligible=True,
        ),
        "new_related_device": Alert(
            alert_id="ALT-DEMO-3",
            alert_type=AlertType.ONLINE_FRAUD,
            customer_account_number="ACC-1001",
            device_id="DEV-NEW-RELATED",
            auto_contact_eligible=False,
        ),
        "company_isp": Alert(
            alert_id="ALT-DEMO-4",
            alert_type=AlertType.ONLINE_FRAUD,
            customer_account_number="ACC-1001",
            device_id="DEV-NEW-SOLO",
            auto_contact_eligible=False,
            metadata={"force_ip": "10.20.30.40"},
        ),
    }
    alert = demos.get(scenario)
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario. Choose one of: {sorted(demos)}",
        )
    return InvestigateResponse(result=orchestrator.run(alert))
