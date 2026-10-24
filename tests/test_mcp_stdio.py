"""End to end over real processes: background API, MCP servers as stdio subprocesses."""

import contextlib
import json

import httpx
import pytest

from securebank.agents import open_mcp_toolsets, stdio_connections
from securebank.cli import background_api
from securebank.client import BankClient
from securebank.tools import ACCOUNT, ACTION, TOOLS_BY_ROLE

pytestmark = pytest.mark.anyio


@pytest.fixture
def api_url(tmp_path, monkeypatch):
    monkeypatch.setenv("SECUREBANK_AUDIT_LOG_PATH", str(tmp_path / "audit.log"))
    with background_api() as url:
        yield url


async def test_stdio_servers_serve_role_tools_for_the_session(api_url):
    client = BankClient.login(httpx.Client(base_url=api_url), "CUST-002")
    connections = stdio_connections(api_url, client.token)

    # The subprocess gets the session token but not this process's secrets.
    assert "ANTHROPIC_API_KEY" not in connections[ACCOUNT]["env"]

    async with contextlib.AsyncExitStack() as stack:
        toolsets = await open_mcp_toolsets(stack, connections)
        assert {t.name for t in toolsets[ACTION]} == set(TOOLS_BY_ROLE[ACTION])

        list_cards = next(t for t in toolsets[ACCOUNT] if t.name == "list_cards")
        [block] = await list_cards.ainvoke({})
        assert [c["card_id"] for c in json.loads(block["text"])] == ["CARD-2001"]
