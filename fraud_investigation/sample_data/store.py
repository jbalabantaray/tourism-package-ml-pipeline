"""In-memory sample SOR data for local demos and tests.

Replace these dicts with real DB/API adapters later — steps stay the same.
"""

from __future__ import annotations

from datetime import datetime, timedelta

NOW = datetime(2026, 10, 27, 9, 0, 0)

DEVICES = {
    "DEV-TRUSTED-001": {
        "device_id": "DEV-TRUSTED-001",
        "customer_account_number": "ACC-1001",
        "first_seen_at": NOW - timedelta(days=200),
        "last_seen_at": NOW - timedelta(hours=2),
        "other_account_ids": [],
    },
    "DEV-NEW-SOLO": {
        "device_id": "DEV-NEW-SOLO",
        "customer_account_number": "ACC-1001",
        "first_seen_at": NOW - timedelta(days=5),
        "last_seen_at": NOW - timedelta(minutes=30),
        "other_account_ids": [],
    },
    "DEV-NEW-RELATED": {
        "device_id": "DEV-NEW-RELATED",
        "customer_account_number": "ACC-1001",
        "first_seen_at": NOW - timedelta(days=10),
        "last_seen_at": NOW - timedelta(hours=1),
        "other_account_ids": ["ACC-2002"],
    },
    "DEV-NEW-UNRELATED": {
        "device_id": "DEV-NEW-UNRELATED",
        "customer_account_number": "ACC-1001",
        "first_seen_at": NOW - timedelta(days=3),
        "last_seen_at": NOW - timedelta(minutes=10),
        "other_account_ids": ["ACC-9999"],
    },
}

ACCOUNTS = {
    "ACC-1001": {
        "account_number": "ACC-1001",
        "last_name": "Sharma",
        "address_line": "12 MG Road",
        "city": "Bengaluru",
        "state": "KA",
        "postal_code": "560001",
        "joint_account_with": ["ACC-2002"],
        "location_label": "Bengaluru, KA",
    },
    "ACC-2002": {
        "account_number": "ACC-2002",
        "last_name": "Sharma",
        "address_line": "12 MG Road",
        "city": "Bengaluru",
        "state": "KA",
        "postal_code": "560001",
        "joint_account_with": ["ACC-1001"],
        "location_label": "Bengaluru, KA",
    },
    "ACC-9999": {
        "account_number": "ACC-9999",
        "last_name": "Patel",
        "address_line": "88 Ring Road",
        "city": "Mumbai",
        "state": "MH",
        "postal_code": "400001",
        "joint_account_with": [],
        "location_label": "Mumbai, MH",
    },
}

IPS = {
    "10.10.1.5": {
        "ip_address": "10.10.1.5",
        "isp": "Airtel Broadband",
        "xa_id": "XA-111",
        "location": "Bengaluru, KA",
        "country": "IN",
        "city": "Bengaluru",
        "first_seen_at": NOW - timedelta(days=180),
        "last_seen_at": NOW - timedelta(hours=1),
        "customer_account_number": "ACC-1001",
    },
    "10.20.30.40": {
        "ip_address": "10.20.30.40",
        "isp": "Microsoft Corporation",
        "xa_id": "XA-222",
        "location": "Hyderabad, TS",
        "country": "IN",
        "city": "Hyderabad",
        "first_seen_at": NOW - timedelta(days=2),
        "last_seen_at": NOW - timedelta(minutes=5),
        "customer_account_number": "ACC-1001",
    },
    "10.55.66.77": {
        "ip_address": "10.55.66.77",
        "isp": "Jio Fiber",
        "xa_id": "XA-333",
        "location": "Bengaluru, KA",
        "country": "IN",
        "city": "Bengaluru",
        "first_seen_at": NOW - timedelta(days=1),
        "last_seen_at": NOW - timedelta(minutes=20),
        "customer_account_number": "ACC-1001",
    },
    "10.88.99.11": {
        "ip_address": "10.88.99.11",
        "isp": "Unknown VPN Hosting",
        "xa_id": "XA-444",
        "location": "Lagos, NG",
        "country": "NG",
        "city": "Lagos",
        "first_seen_at": NOW - timedelta(hours=6),
        "last_seen_at": NOW - timedelta(minutes=3),
        "customer_account_number": "ACC-1001",
    },
}

# OAT events keyed to devices so demos are deterministic
OAT_EVENTS = [
    {
        "event_id": "EVT-TRUSTED",
        "customer_account_number": "ACC-1001",
        "device_id": "DEV-TRUSTED-001",
        "ip_address": "10.10.1.5",
        "login_at": NOW - timedelta(minutes=20),
        "suspicious": True,
        "alerted": True,
        "xa_id": "XA-111",
        "location": "Bengaluru, KA",
        "isp": "Airtel Broadband",
    },
    {
        "event_id": "EVT-SOLO-CORP",
        "customer_account_number": "ACC-1001",
        "device_id": "DEV-NEW-SOLO",
        "ip_address": "10.20.30.40",
        "login_at": NOW - timedelta(minutes=15),
        "suspicious": True,
        "alerted": True,
        "xa_id": "XA-222",
        "location": "Hyderabad, TS",
        "isp": "Microsoft Corporation",
    },
    {
        "event_id": "EVT-RELATED",
        "customer_account_number": "ACC-1001",
        "device_id": "DEV-NEW-RELATED",
        "ip_address": "10.55.66.77",
        "login_at": NOW - timedelta(minutes=12),
        "suspicious": True,
        "alerted": True,
        "xa_id": "XA-333",
        "location": "Bengaluru, KA",
        "isp": "Jio Fiber",
    },
    {
        "event_id": "EVT-UNRELATED",
        "customer_account_number": "ACC-1001",
        "device_id": "DEV-NEW-UNRELATED",
        "ip_address": "10.88.99.11",
        "login_at": NOW - timedelta(minutes=10),
        "suspicious": True,
        "alerted": True,
        "xa_id": "XA-444",
        "location": "Lagos, NG",
        "isp": "Unknown VPN Hosting",
    },
]

INTERACTION_NOTES = [
    {
        "note_id": "NOTE-1",
        "customer_account_number": "ACC-1001",
        "created_at": NOW - timedelta(days=40),
        "channel": "phone",
        "text": "Customer confirmed travel to Delhi next month.",
        "verified_activity": True,
    }
]

COMPANY_ISP_KEYWORDS = [
    "microsoft",
    "google",
    "amazon",
    "apple",
    "meta",
    "oracle",
    "salesforce",
    "accenture",
    "infosys",
    "tcs",
    "wipro",
]
