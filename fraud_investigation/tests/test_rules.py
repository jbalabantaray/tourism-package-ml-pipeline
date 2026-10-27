from datetime import datetime, timedelta

from fraud_investigation.rules.device_rules import evaluate_device
from fraud_investigation.rules.ip_rules import evaluate_ip, is_company_isp
from fraud_investigation.rules.relationship import accounts_related
from fraud_investigation.sample_data import store as sample


def test_relationship_joint_and_same_name():
    payer = sample.ACCOUNTS["ACC-1001"]
    related = sample.ACCOUNTS["ACC-2002"]
    unrelated = sample.ACCOUNTS["ACC-9999"]
    ok, reasons = accounts_related(payer, related)
    assert ok
    assert "joint_account" in reasons or "same_last_name" in reasons
    bad, _ = accounts_related(payer, unrelated)
    assert not bad


def test_company_isp_microsoft():
    assert is_company_isp("Microsoft Corporation", sample.COMPANY_ISP_KEYWORDS)
    assert not is_company_isp("Airtel Broadband", sample.COMPANY_ISP_KEYWORDS)


def test_device_tenure_green():
    device = sample.DEVICES["DEV-TRUSTED-001"]
    indicators = evaluate_device(device, sample.ACCOUNTS["ACC-1001"], [], as_of=sample.NOW)
    assert indicators[0].code == "DEVICE_TENURE_OK"
    assert indicators[0].flag.value == "GREEN"


def test_device_unrelated_red():
    device = sample.DEVICES["DEV-NEW-UNRELATED"]
    others = [sample.ACCOUNTS["ACC-9999"]]
    indicators = evaluate_device(
        device, sample.ACCOUNTS["ACC-1001"], others, as_of=sample.NOW
    )
    assert indicators[0].code == "DEVICE_UNRELATED_COUSER"
    assert indicators[0].flag.value == "RED"


def test_ip_location_risk():
    ip = sample.IPS["10.88.99.11"]
    indicators = evaluate_ip(
        ip,
        sample.ACCOUNTS["ACC-1001"],
        sample.COMPANY_ISP_KEYWORDS,
        as_of=sample.NOW,
    )
    assert indicators[0].code == "IP_LOCATION_RISK"


def test_ip_company_isp_green():
    ip = sample.IPS["10.20.30.40"]
    indicators = evaluate_ip(
        ip,
        sample.ACCOUNTS["ACC-1001"],
        sample.COMPANY_ISP_KEYWORDS,
        as_of=sample.NOW,
    )
    assert indicators[0].code == "IP_COMPANY_ISP"
