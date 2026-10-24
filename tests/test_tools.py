"""Bank tools as agents see them: served per role by the MCP server, over an MCP session."""

import json
from datetime import date

import pytest
from fakes import mcp_toolsets
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from securebank.api import create_app
from securebank.audit import AuditLog
from securebank.client import BankAPIError, BankClient
from securebank.data import seed_store
from securebank.mcp_server import ALL, create_mcp_server
from securebank.tools import ACCOUNT, ACTION, FRAUD, POLICY, TOOLS_BY_ROLE, WRITE_TOOLS

TODAY = date(2026, 10, 22)

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path):
    return create_app(seed_store(TODAY), AuditLog(tmp_path / "audit.log"))


async def toolsets_for(app, customer_id):
    client = BankClient.login(TestClient(app), customer_id)
    async with mcp_toolsets(client) as sets:
        yield {role: {t.name: t for t in tools} for role, tools in sets.items()}


@pytest.fixture
async def alex(app):
    async for sets in toolsets_for(app, "CUST-001"):
        yield sets


@pytest.fixture
async def sam(app):
    async for sets in toolsets_for(app, "CUST-002"):
        yield sets


async def run(tool, **args) -> str:
    out = await tool.ainvoke(args)
    return out if isinstance(out, str) else out[0]["text"] if isinstance(out, list) else str(out)


def test_login_unknown_customer(app):
    with pytest.raises(BankAPIError) as exc:
        BankClient.login(TestClient(app), "CUST-999")
    assert exc.value.status_code == 401


def test_from_token_joins_existing_session(app):
    first = BankClient.login(TestClient(app), "CUST-002")
    joined = BankClient.from_token(TestClient(app), first.token)
    assert joined.customer_id == "CUST-002"
    with pytest.raises(BankAPIError):
        BankClient.from_token(TestClient(app), "bogus")


async def test_each_role_server_exposes_only_its_tools(alex):
    for role, names in TOOLS_BY_ROLE.items():
        assert set(alex[role]) == set(names)
    for role in (ACCOUNT, FRAUD, POLICY):
        assert not WRITE_TOOLS & alex[role].keys()


async def test_all_role_exposes_every_tool(app):
    client = BankClient.login(TestClient(app), "CUST-001")
    server = create_mcp_server(client, ALL)
    async with create_connected_server_and_client_session(server) as session:
        names = {t.name for t in (await session.list_tools()).tools}
    assert names == {n for tools in TOOLS_BY_ROLE.values() for n in tools}


async def test_no_tool_accepts_a_customer_id(alex):
    for tools in alex.values():
        for t in tools.values():
            assert "customer_id" not in t.args


async def test_account_tools(alex):
    cards = json.loads(await run(alex[ACCOUNT]["list_cards"]))
    assert {c["card_id"] for c in cards} == {"CARD-1001", "CARD-1002"}
    [dispute] = json.loads(await run(alex[ACCOUNT]["list_disputes"]))
    assert dispute["dispute_id"] == "DSP-0001"


async def test_other_customers_data_comes_back_as_error_text(sam):
    out = await run(sam[ACCOUNT]["list_transactions"], card_id="CARD-1001")
    assert out.startswith("Error (404)")


async def test_fraud_signals_tool(alex):
    signals = json.loads(await run(alex[FRAUD]["find_fraud_signals"], card_id="CARD-1001"))
    found = {(s["transaction_id"], s["kind"]) for s in signals}
    assert ("TXN-0004", "possible_duplicate") in found
    assert ("TXN-0005", "foreign") in found
    assert ("TXN-0005", "unusually_large") in found


@pytest.mark.parametrize(
    ("transaction_id", "reason", "eligible", "problem"),
    [
        ("TXN-0005", "fraud", True, None),
        ("TXN-0006", "fraud", False, "within 120 days"),
        ("TXN-0011", "incorrect_amount", False, "Pending"),
        ("TXN-0008", "not_received", False, "DSP-0001"),
    ],
)
async def test_check_dispute_eligibility(alex, transaction_id, reason, eligible, problem):
    out = json.loads(
        await run(
            alex[POLICY]["check_dispute_eligibility"],
            transaction_id=transaction_id,
            reason=reason,
        )
    )
    assert out["eligible"] is eligible
    if problem:
        assert any(problem in p for p in out["problems"])


async def test_get_dispute_policy(alex):
    assert len(json.loads(await run(alex[POLICY]["get_dispute_policy"]))) == 4
    [rule] = json.loads(await run(alex[POLICY]["get_dispute_policy"], reason="duplicate"))
    assert rule["window_days"] == 60


async def test_block_card(alex):
    out = json.loads(await run(alex[ACTION]["block_card"], card_id="CARD-1001", reason="lost"))
    assert out["status"] == "blocked"


async def test_file_dispute(alex):
    out = json.loads(
        await run(
            alex[ACTION]["file_dispute"],
            transaction_id="TXN-0004",
            reason="duplicate",
            description="Charged twice.",
        )
    )
    assert out["transaction_id"] == "TXN-0004"
