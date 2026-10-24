import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from securebank.api import create_app
from securebank.audit import AuditLog
from securebank.data import seed_store

TODAY = date(2026, 10, 22)


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.log")


@pytest.fixture
def client(audit):
    return TestClient(create_app(seed_store(TODAY), audit))


def login(client, customer_id):
    resp = client.post("/auth/session", json={"customer_id": customer_id})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['token']}"}


@pytest.fixture
def alex(client):
    return login(client, "CUST-001")


@pytest.fixture
def sam(client):
    return login(client, "CUST-002")


def test_login_unknown_customer(client):
    assert client.post("/auth/session", json={"customer_id": "CUST-999"}).status_code == 401


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-token"}])
def test_requires_valid_token(client, headers):
    assert client.get("/cards", headers=headers).status_code == 401


def test_me(client, alex):
    assert client.get("/me", headers=alex).json()["customer_id"] == "CUST-001"


def test_lists_only_own_cards(client, alex, sam):
    assert {c["card_id"] for c in client.get("/cards", headers=alex).json()} == {
        "CARD-1001",
        "CARD-1002",
    }
    assert [c["card_id"] for c in client.get("/cards", headers=sam).json()] == ["CARD-2001"]


@pytest.mark.parametrize(
    "path",
    [
        "/cards/CARD-1001",
        "/cards/CARD-1001/transactions",
        "/transactions/TXN-0005",
        "/disputes/DSP-0001",
    ],
)
def test_other_customers_records_look_missing(client, sam, path):
    resp = client.get(path, headers=sam)
    assert resp.status_code == 404
    missing = client.get(path.replace("1001", "9999").replace("0005", "9999"), headers=sam)
    assert missing.status_code == 404


def test_transactions(client, alex):
    txns = client.get("/cards/CARD-1001/transactions", headers=alex).json()
    assert {t["transaction_id"] for t in txns} >= {"TXN-0003", "TXN-0004", "TXN-0005"}
    assert client.get("/transactions/TXN-0005", headers=alex).json()["country"] == "LT"


def test_block_card(client, alex, sam):
    assert client.post("/cards/CARD-1001/block", headers=sam).status_code == 404
    resp = client.post("/cards/CARD-1001/block", headers=alex)
    assert resp.status_code == 200
    assert resp.json()["status"] == "blocked"


def test_create_dispute(client, alex):
    body = {"transaction_id": "TXN-0004", "reason": "duplicate", "description": "Charged twice."}
    resp = client.post("/disputes", json=body, headers=alex)
    assert resp.status_code == 201
    assert resp.json()["customer_id"] == "CUST-001"
    assert {d["dispute_id"] for d in client.get("/disputes", headers=alex).json()} == {
        "DSP-0001",
        resp.json()["dispute_id"],
    }
    assert client.post("/disputes", json=body, headers=alex).status_code == 409


def test_cannot_dispute_other_customers_transaction(client, sam):
    body = {"transaction_id": "TXN-0005", "reason": "fraud", "description": "Not me."}
    assert client.post("/disputes", json=body, headers=sam).status_code == 404


@pytest.mark.parametrize(
    ("transaction_id", "reason"),
    [("TXN-0006", "fraud"), ("TXN-0011", "incorrect_amount")],  # too old; still pending
)
def test_api_enforces_dispute_policy(client, alex, transaction_id, reason):
    body = {"transaction_id": transaction_id, "reason": reason, "description": "x"}
    assert client.post("/disputes", json=body, headers=alex).status_code == 422


def test_fraud_signals(client, alex, sam):
    signals = client.get("/cards/CARD-1001/fraud-signals", headers=alex).json()
    assert {(s["transaction_id"], s["kind"]) for s in signals} == {
        ("TXN-0004", "possible_duplicate"),
        ("TXN-0005", "foreign"),
        ("TXN-0005", "unusually_large"),
    }
    assert client.get("/cards/CARD-1001/fraud-signals", headers=sam).status_code == 404


def test_dispute_policy(client, alex):
    assert client.get("/policy/disputes").status_code == 401
    assert len(client.get("/policy/disputes", headers=alex).json()) == 4
    resp = client.get("/policy/disputes", params={"reason": "fraud"}, headers=alex)
    assert [r["window_days"] for r in resp.json()] == [120]


def test_eligibility(client, alex, sam):
    path = "/transactions/TXN-0008/eligibility"
    resp = client.get(path, params={"reason": "not_received"}, headers=alex).json()
    assert resp == {"eligible": False, "problems": ["Already has active dispute DSP-0001."]}
    resp = client.get(
        "/transactions/TXN-0005/eligibility", params={"reason": "fraud"}, headers=alex
    )
    assert resp.json() == {"eligible": True, "problems": []}
    assert client.get(path, params={"reason": "fraud"}, headers=sam).status_code == 404


def test_dispute_validation(client, alex):
    body = {"transaction_id": "TXN-0005", "reason": "bogus", "description": "x"}
    assert client.post("/disputes", json=body, headers=alex).status_code == 422


def audit_entries(audit):
    return [json.loads(line) for line in audit.path.read_text().splitlines()]


def test_actions_are_audited(client, audit, alex):
    client.post("/cards/CARD-1001/block", json={"reason": "lost card"}, headers=alex)
    body = {"transaction_id": "TXN-0004", "reason": "duplicate", "description": "Charged twice."}
    client.post("/disputes", json=body, headers=alex)
    block, dispute = audit_entries(audit)
    assert block["customer_id"] == "CUST-001"
    assert block["action"] == "block_card"
    assert block["args"] == {"card_id": "CARD-1001", "reason": "lost card"}
    assert block["outcome"] == "ok"
    assert dispute["action"] == "file_dispute"
    assert dispute["args"] == body


def test_block_without_reason(client, audit, alex):
    assert client.post("/cards/CARD-1002/block", headers=alex).status_code == 200
    assert audit_entries(audit)[0]["args"]["reason"] is None


def test_rejected_actions_are_audited(client, audit, alex, sam):
    client.post("/cards/CARD-1001/block", headers=sam)
    body = {"transaction_id": "TXN-0006", "reason": "fraud", "description": "x"}
    client.post("/disputes", json=body, headers=alex)
    denied, ineligible = audit_entries(audit)
    assert (denied["customer_id"], denied["outcome"], denied["status"]) == (
        "CUST-002",
        "error",
        404,
    )
    assert (ineligible["outcome"], ineligible["status"]) == ("error", 422)


def test_reads_are_not_audited(client, audit, alex):
    client.get("/cards", headers=alex)
    assert not audit.path.exists()
