from datetime import date

import pytest
from fakes import ScriptedModel, call, make_graph, mcp_toolsets
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from securebank import cli
from securebank.api import create_app
from securebank.audit import AuditLog
from securebank.client import BankClient
from securebank.config import Settings
from securebank.data import seed_store
from securebank.models import CardStatus, DisputeStatus

TODAY = date(2026, 10, 22)

pytestmark = pytest.mark.anyio


@pytest.fixture
def store():
    return seed_store(TODAY)


@pytest.fixture
def client(store, tmp_path):
    app = create_app(store, AuditLog(tmp_path / "audit.log"))
    return BankClient.login(TestClient(app), "CUST-001")


@pytest.fixture
async def toolsets(client):
    async with mcp_toolsets(client) as sets:
        yield sets


async def run(graph, client, inputs):
    """Drive run_chat with scripted input; return everything it printed and prompted."""
    inputs = iter(inputs)
    transcript = []

    def read(prompt):
        transcript.append(prompt)
        return next(inputs)

    await cli.run_chat(graph, client, read=read, write=transcript.append)
    return "\n".join(transcript)


def block_card_graph(toolsets):
    supervisor = ScriptedModel(
        replies=[
            call("ask_action_agent", "s1", task="Block CARD-1001; customer reports it lost."),
            AIMessage("Your credit card is now blocked."),
        ],
        calls=[],
    )
    specialist = ScriptedModel(
        replies=[
            call("block_card", "a1", card_id="CARD-1001", reason="lost"),
            AIMessage("Blocked CARD-1001."),
        ],
        calls=[],
    )
    return make_graph(toolsets, supervisor, specialist)


async def test_confirmed_action(client, toolsets, store):
    out = await run(block_card_graph(toolsets), client, ["I lost my card, block it", "y", "quit"])

    assert "Signed in as Alex Rivera" in out
    # The confirmation is described from bank records, not the model's task text.
    assert "Confirm: Block your credit card ending 4242." in out
    assert "bank> Your credit card is now blocked." in out
    assert store.get_card("CARD-1001").status == CardStatus.BLOCKED


@pytest.mark.parametrize("answer", ["n", "", "maybe"])
async def test_anything_but_yes_declines(client, toolsets, store, answer):
    await run(block_card_graph(toolsets), client, ["block my card", answer, "quit"])
    assert store.get_card("CARD-1001").status == CardStatus.ACTIVE


async def test_two_actions_in_one_turn(client, toolsets, store):
    supervisor = ScriptedModel(
        replies=[
            call("ask_action_agent", "s1", task="Block CARD-1001 and dispute TXN-0005 as fraud."),
            AIMessage("Done."),
        ],
        calls=[],
    )
    both = AIMessage(
        "",
        tool_calls=[
            {"name": "block_card", "args": {"card_id": "CARD-1001", "reason": "fraud"}, "id": "a1"},
            {
                "name": "file_dispute",
                "args": {
                    "transaction_id": "TXN-0005",
                    "reason": "fraud",
                    "description": "I never shopped in Vilnius.",
                },
                "id": "a2",
            },
        ],
    )
    specialist = ScriptedModel(replies=[both, AIMessage("Both done.")], calls=[])
    graph = make_graph(toolsets, supervisor, specialist)

    out = await run(graph, client, ["that Vilnius charge isn't mine", "y", "y", "quit"])

    assert "File a fraud dispute on LUXGOODS VILNIUS, 1899.00 USD" in out
    assert store.get_card("CARD-1001").status == CardStatus.BLOCKED
    [dispute] = [d for d in store.list_disputes("CUST-001") if d.transaction_id == "TXN-0005"]
    assert dispute.status == DisputeStatus.OPEN


async def test_eof_exits_cleanly(client, toolsets):
    graph = make_graph(
        toolsets,
        ScriptedModel(replies=[], calls=[]),
        ScriptedModel(replies=[], calls=[]),
    )

    def read(prompt):
        raise EOFError

    await cli.run_chat(graph, client, read=read, write=lambda s: None)


def test_chat_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    assert cli.main(["chat"]) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_no_command_prints_status(capsys):
    assert cli.main([]) == 0
    assert "SecureBank v" in capsys.readouterr().out
