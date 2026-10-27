from fraud_investigation.models.schemas import Alert, AlertType, DispositionType
from fraud_investigation.orchestrator.engine import InvestigationOrchestrator


orch = InvestigationOrchestrator()


def test_trusted_device_no_fraud():
    alert = Alert(
        alert_id="T1",
        alert_type=AlertType.ONLINE_FRAUD,
        customer_account_number="ACC-1001",
        device_id="DEV-TRUSTED-001",
        auto_contact_eligible=False,
    )
    result = orch.run(alert)
    assert result.disposition == DispositionType.NO_FRAUD
    assert result.contact_customer is False
    assert any(g.code == "DEVICE_TENURE_OK" for g in result.green_flags)


def test_unrelated_device_pending_contact():
    alert = Alert(
        alert_id="T2",
        alert_type=AlertType.ONLINE_FRAUD,
        customer_account_number="ACC-1001",
        device_id="DEV-NEW-UNRELATED",
        auto_contact_eligible=True,
    )
    result = orch.run(alert)
    assert result.disposition == DispositionType.PENDING
    assert result.contact_customer is True
    assert any(i.code == "DEVICE_UNRELATED_COUSER" for i in result.risk_indicators)
    assert any(i.code == "IP_LOCATION_RISK" for i in result.risk_indicators)


def test_company_isp_path():
    alert = Alert(
        alert_id="T3",
        alert_type=AlertType.ONLINE_FRAUD,
        customer_account_number="ACC-1001",
        device_id="DEV-NEW-SOLO",
        auto_contact_eligible=False,
    )
    result = orch.run(alert)
    assert any(g.code == "IP_COMPANY_ISP" for g in result.green_flags)
    # New solo device is green; company ISP green -> no reds expected
    assert result.disposition == DispositionType.NO_FRAUD
