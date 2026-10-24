"""Test doubles shared across test modules."""

import contextlib

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from securebank.agents import build_supervisor
from securebank.mcp_server import create_mcp_server
from securebank.tools import ROLES


class ScriptedModel(BaseChatModel):
    """Returns queued replies in order and records the messages it was sent."""

    replies: list[AIMessage]
    calls: list[list] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def call(name, id_, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": id_}])


def make_graph(toolsets, supervisor, specialist):
    models = iter([specialist] * 4 + [supervisor])  # build order: 4 specialists, then supervisor
    return build_supervisor(toolsets, lambda effort: next(models))


@contextlib.asynccontextmanager
async def mcp_toolsets(client):
    """Load each role's tools from its own MCP server over an in-memory session."""
    async with contextlib.AsyncExitStack() as stack:
        toolsets = {}
        for role in ROLES:
            server = create_mcp_server(client, role)
            session = await stack.enter_async_context(
                create_connected_server_and_client_session(server)
            )
            toolsets[role] = await load_mcp_tools(session)
        yield toolsets
