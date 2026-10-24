"""Graph wiring tests with a scripted fake model, so no API calls are made."""

from datetime import date

import pytest
from fakes import ScriptedModel, call, make_graph, mcp_toolsets
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import Command

from securebank.api import create_app
from securebank.audit import AuditLog
from securebank.client import BankClient
from securebank.data import seed_store
from securebank.models import CardStatus

TODAY = date(2026, 10, 22)

pytestmark = pytest.mark.anyio


@pytest.fixture
def store():
    return seed_store(TODAY)


@pytest.fixture
async def toolsets(store, tmp_path):
    app = create_app(store, AuditLog(tmp_path / "audit.log"))
    async with mcp_toolsets(BankClient.login(TestClient(app), "CUST-001")) as sets:
        yield sets


CONFIG = {"configurable": {"thread_id": "t1"}}


async def test_delegates_to_account_agent(toolsets):
    supervisor = ScriptedModel(
        replies=[
            call("ask_account_agent", "s1", task="List the customer's cards."),
            AIMessage("You have a credit card ending 4242 and a debit card ending 1881."),
        ],
        calls=[],
    )
    specialist = ScriptedModel(
        replies=[call("list_cards", "a1"), AIMessage("Cards: CARD-1001 (4242), CARD-1002.")],
        calls=[],
    )
    graph = make_graph(toolsets, supervisor, specialist)

    result = await graph.ainvoke({"messages": [("user", "What cards do I have?")]}, CONFIG)

    assert "4242" in result["messages"][-1].text
    # The specialist got a fresh conversation: just the supervisor's task.
    first_specialist_call = specialist.calls[0]
    assert [m.type for m in first_specialist_call] == ["system", "human"]
    assert first_specialist_call[1].content == "List the customer's cards."
    # It saw the real tool output from the API.
    assert "CARD-1001" in specialist.calls[1][-1].text


def action_graph(toolsets):
    supervisor = ScriptedModel(
        replies=[
            call("ask_action_agent", "s1", task="Block card CARD-1001; customer reports it lost."),
            AIMessage("Done."),
        ],
        calls=[],
    )
    specialist = ScriptedModel(
        replies=[
            call("block_card", "a1", card_id="CARD-1001", reason="lost"),
            AIMessage("Result reported."),
        ],
        calls=[],
    )
    return make_graph(toolsets, supervisor, specialist), specialist


async def test_action_waits_for_confirmation_then_runs(toolsets, store):
    graph, specialist = action_graph(toolsets)

    result = await graph.ainvoke({"messages": [("user", "Block my credit card")]}, CONFIG)

    [pending] = result["__interrupt__"]
    assert pending.value == {
        "action": "block_card",
        "args": {"card_id": "CARD-1001", "reason": "lost"},
    }
    assert store.get_card("CARD-1001").status == CardStatus.ACTIVE

    await graph.ainvoke(Command(resume=True), CONFIG)

    assert store.get_card("CARD-1001").status == CardStatus.BLOCKED
    assert '"status": "blocked"' in specialist.calls[-1][-1].text
    # Resuming continued the action agent from its checkpoint instead of re-asking the model.
    assert len(specialist.calls) == 2


async def test_declined_action_changes_nothing(toolsets, store):
    graph, specialist = action_graph(toolsets)
    await graph.ainvoke({"messages": [("user", "Block my credit card")]}, CONFIG)

    await graph.ainvoke(Command(resume=False), CONFIG)

    assert store.get_card("CARD-1001").status == CardStatus.ACTIVE
    assert "declined" in specialist.calls[-1][-1].text


async def test_supervisor_has_only_delegate_tools(toolsets):
    graph = make_graph(
        toolsets, ScriptedModel(replies=[], calls=[]), ScriptedModel(replies=[], calls=[])
    )
    tool_node = graph.get_graph().nodes["tools"].data
    assert set(tool_node.tools_by_name) == {
        "ask_account_agent",
        "ask_fraud_agent",
        "ask_policy_agent",
        "ask_action_agent",
    }
