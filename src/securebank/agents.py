"""LangGraph agents: a supervisor that delegates to account, fraud, policy, and action agents.

Each specialist is exposed to the supervisor as a tool and runs in a fresh conversation built
from the supervisor's task description. Only the supervisor's conversation persists across
turns, and it is append-only. Agents never share a message history, so no agent sees another
agent's tools or replays thinking blocks produced under a different system prompt.

Specialists get their tools over MCP, one server session per role, so each agent can only
call what its role's server exposes. Action tools pause the graph with a LangGraph interrupt so
the customer confirms every change outside the model. The caller resumes with
Command(resume=...) to proceed or cancel. MCP tools are async, so run the graph with ainvoke.
"""

import sys
from collections.abc import Callable
from contextlib import AsyncExitStack

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from mcp.client.stdio import get_default_environment

from securebank.config import Settings
from securebank.mcp_server import TOKEN_ENV
from securebank.tools import ACCOUNT, ACTION, FRAUD, POLICY, ROLES

# Models that accept the server-side refusal fallback in its "default" form.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

SUPERVISOR_PROMPT = """\
You are SecureBank's card dispute assistant, talking with a signed-in customer. You help them \
understand their cards and transactions, spot fraud, check dispute rules, and block cards or \
file disputes.

You have no direct access to bank systems. Delegate to specialists, giving each a \
self-contained task with every ID and detail it needs, since they cannot see this conversation:
- account: profile, cards, transactions, existing disputes.
- fraud: reviews a card's transactions for duplicates and suspicious charges.
- policy: dispute rules, filing windows, and whether a transaction is eligible.
- action: blocks a card or files a dispute.

Before delegating an action, check eligibility with policy, explain to the customer what will \
happen, and make sure they want it. The system then asks the customer to confirm the exact \
change; if they decline, nothing changes. For suspected fraud, recommend blocking the card \
before disputing.

Refer to cards by type and last four digits rather than internal IDs. Never invent IDs, \
amounts, or outcomes; if a specialist reports an error, tell the customer plainly. You only \
handle this customer's cards and disputes; decline unrelated requests briefly."""

SPECIALIST_PROMPTS = {
    ACCOUNT: (
        "You look up the signed-in customer's profile, cards, transactions, and disputes. "
        "Answer the task with the specific records found, including IDs, dates, amounts, and "
        "statuses. If a record isn't found, say so."
    ),
    FRAUD: (
        "You review the signed-in customer's card activity for fraud. Use find_fraud_signals "
        "and the transaction history, then report which transactions look suspicious and why, "
        "how confident you are, and what you recommend. Signals are evidence, not proof: weigh "
        "them against anything the task says the customer told us."
    ),
    POLICY: (
        "You answer questions about card dispute rules. Look up the rules and check "
        "eligibility for specific transactions rather than relying on memory. Report whether "
        "each transaction is eligible and, if not, exactly why."
    ),
    ACTION: (
        "You carry out changes on the signed-in customer's account: blocking cards and filing "
        "disputes. Do exactly what the task asks, using the IDs it gives. Each change asks the "
        "customer to confirm; if they decline, don't retry. Report what was done, including "
        "new dispute IDs, or why it failed."
    ),
}

DELEGATE_DESCRIPTIONS = {
    ACCOUNT: "Ask the account specialist to look up the customer's profile, cards, "
    "transactions, or disputes.",
    FRAUD: "Ask the fraud specialist to review a card's transactions for suspicious activity.",
    POLICY: "Ask the policy specialist about dispute rules or whether a transaction can be "
    "disputed.",
    ACTION: "Ask the action specialist to block a card or file a dispute. Only after the "
    "customer has agreed.",
}

# Lookups and executing explicit instructions need little reasoning; judgment calls get more.
SPECIALIST_EFFORT = {ACCOUNT: "low", ACTION: "low"}


def build_llm(settings: Settings, effort: str | None = None) -> ChatAnthropic:
    kwargs = {}
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    if settings.model in _FALLBACK_MODELS:
        # If a safety classifier declines, the API reruns the request on a fallback model it
        # picks by refusal category, within the same call.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["model_kwargs"] = {"fallbacks": "default"}
    return ChatAnthropic(
        model=settings.model,
        reasoning_effort=effort or settings.effort,
        max_tokens=settings.max_tokens,
        **kwargs,
    )


def require_confirmation(action_tool: BaseTool) -> BaseTool:
    """Wrap an action tool so it runs only after the customer confirms via an interrupt."""

    async def confirmed(**kwargs):
        approved = interrupt({"action": action_tool.name, "args": kwargs})
        if not approved:
            return "The customer declined. Nothing was changed."
        return await action_tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        coroutine=confirmed,
        name=action_tool.name,
        description=action_tool.description,
        args_schema=action_tool.args_schema,
    )


def _delegate(name: str, agent: CompiledStateGraph) -> BaseTool:
    @tool(f"ask_{name}_agent", description=DELEGATE_DESCRIPTIONS[name])
    async def ask(task: str) -> str:
        result = await agent.ainvoke({"messages": [HumanMessage(task)]})
        return result["messages"][-1].text or "(no answer)"

    return ask


def stdio_connections(api_url: str, token: str) -> dict[str, dict]:
    """One MCP server subprocess per role, joined to the customer's existing API session.

    The subprocess gets a minimal environment plus the session token, not this process's
    environment, so secrets such as the Anthropic API key stay here.
    """
    env = {**get_default_environment(), TOKEN_ENV: token}
    return {
        role: {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "securebank.mcp_server", "--role", role, "--api-url", api_url],
            "env": env,
        }
        for role in ROLES
    }


async def open_mcp_toolsets(
    stack: AsyncExitStack, connections: dict[str, dict]
) -> dict[str, list[BaseTool]]:
    """Open a persistent MCP session per role and load its tools. Sessions close with `stack`."""
    client = MultiServerMCPClient(connections)
    toolsets = {}
    for role in ROLES:
        session = await stack.enter_async_context(client.session(role))
        toolsets[role] = await load_mcp_tools(session)
    return toolsets


def build_supervisor(
    toolsets: dict[str, list[BaseTool]],
    llm_for: Callable[[str | None], BaseChatModel],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the supervisor graph.

    `llm_for(effort)` returns a chat model at the given effort level (None for the default),
    so tests can substitute a fake model.
    """
    delegates = []
    for name in ROLES:
        tools = toolsets[name]
        if name == ACTION:
            tools = [require_confirmation(t) for t in tools]
        agent = create_agent(
            llm_for(SPECIALIST_EFFORT.get(name)),
            tools,
            system_prompt=SPECIALIST_PROMPTS[name],
            name=f"{name}_agent",
        )
        delegates.append(_delegate(name, agent))

    return create_agent(
        llm_for(None),
        delegates,
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
        name="supervisor",
    )
