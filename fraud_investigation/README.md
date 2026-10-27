# Fraud Investigation Service (sample)

Python + FastAPI greenfield scaffold that automates online fraud alert investigation.

- **Deterministic rules** for device tenure, multi-user relationship, IP tenure, company ISP, geo sense
- **Pluggable SOR adapters** (in-memory today; swap to DB/HTTP via config)
- **Config-driven process flow** (`processes/online_fraud_v1.yaml`)
- **LLM hook** stub for unstructured notes/location interpretation

## Quick start

```bash
cd /workspace
python -m venv .venv
source .venv/bin/activate
pip install -r fraud_investigation/requirements.txt
pip install pytest
export PYTHONPATH=/workspace

# Run API
uvicorn fraud_investigation.api.main:app --reload --port 8080

# Run tests
pytest fraud_investigation/tests -q
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/processes` | Listed process definitions |
| POST | `/investigate` | Investigate an alert payload |
| POST | `/investigate/demo/{scenario}` | Canned demos |

Demo scenarios: `trusted_device`, `new_unrelated_device`, `new_related_device`, `company_isp`

### Example

```bash
curl -s http://127.0.0.1:8080/investigate/demo/new_unrelated_device | python -m json.tool
```

```bash
curl -s http://127.0.0.1:8080/investigate \
  -H 'Content-Type: application/json' \
  -d '{
    "alert": {
      "alert_id": "ALT-100",
      "alert_type": "ONLINE_FRAUD",
      "customer_account_number": "ACC-1001",
      "device_id": "DEV-NEW-UNRELATED",
      "auto_contact_eligible": true
    }
  }'
```

## Alert schema

- `alert_type`
- `customer_account_number`
- `device_id`
- `alert_id`
- `auto_contact_eligible`

## Disposition types

`Fraud` | `No fraud` | `Pending`

## OAT rules encoded

**Device**
1. Tenure > 90 days → No risk
2. Else no other users → No risk
3. Else other users related to payer (same address / last name / location / joint account) → No risk
4. Else → Risk indicator

**IP** (SOR fields: device id context, ISP, XA ID, location)
1. Company ISP (e.g. name contains `Microsoft`) → No fraud
2. IP tenure > 90 days → No fraud
3. Else location makes sense vs customer profile → No risk
4. Else → Risk indicator

## Adding a new investigation process

1. Add SOR adapter under `adapters/` implementing `SORAdapter.fetch`
2. Register it in `orchestrator/registry.py`
3. Add step under `steps/` (or reuse) + register
4. Add/extend YAML in `processes/`
5. Point `adapters:` keys at DB or HTTP concrete adapters

No orchestrator rewrite required.

## Layout

```text
fraud_investigation/
  api/                 # FastAPI
  orchestrator/        # YAML loader + engine
  adapters/            # SOR plugins (memory / future HTTP-DB)
  steps/               # investigation units
  rules/               # deterministic decision trees
  llm/                 # optional interpreter stub
  processes/           # YAML process definitions
  sample_data/         # demo fixtures
  tests/
```
