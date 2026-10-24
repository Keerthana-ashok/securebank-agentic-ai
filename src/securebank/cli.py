"""Command-line entry point: chat with the assistant, run the API server, or run an MCP server."""

import argparse
import asyncio
import contextlib
import socket
import sys
import threading
import uuid
from collections.abc import Callable, Iterator

import anthropic
import httpx
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from securebank import __version__
from securebank.client import BankAPIError, BankClient
from securebank.config import Settings, get_settings

QUIT_WORDS = {"quit", "exit", "bye"}


def describe_action(client: BankClient, action: str, args: dict) -> str:
    """Describe a pending action from the bank's own records, not the model's wording."""
    try:
        if action == "block_card":
            card = next(c for c in client.list_cards() if c.card_id == args["card_id"])
            return f"Block your {card.card_type} card ending {card.last4}."
        if action == "file_dispute":
            txn = client.get_transaction(args["transaction_id"])
            return (
                f"File a {args['reason'].replace('_', ' ')} dispute on {txn.merchant}, "
                f"{txn.amount} {txn.currency} on {txn.posted_on}.\n"
                f"  Your description: {args['description']}"
            )
    except (BankAPIError, StopIteration, KeyError):
        pass  # fall through; the API will reject it when it runs
    return f"{action} with {args}"


async def run_chat(
    graph: CompiledStateGraph,
    client: BankClient,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    customer = client.me()
    write(f"Signed in as {customer.name}. Ask about your cards, or type 'quit' to leave.\n")

    while True:
        try:
            text = (await asyncio.to_thread(read, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            return
        if not text:
            continue
        if text.lower() in QUIT_WORDS:
            return

        try:
            result = await graph.ainvoke({"messages": [("user", text)]}, config)
            while pending := result.get("__interrupt__"):
                decisions = {}
                for item in pending:
                    write("\nConfirm: " + describe_action(client, **item.value))
                    answer = (await asyncio.to_thread(read, "Proceed? [y/N] ")).strip().lower()
                    decisions[item.id] = answer in {"y", "yes"}
                result = await graph.ainvoke(Command(resume=decisions), config)
        except anthropic.APIConnectionError:
            write("bank> Couldn't reach the assistant service. Check your connection and retry.\n")
            continue
        except anthropic.APIStatusError as e:
            write(f"bank> The assistant service returned an error ({e.status_code}). Try again.\n")
            continue

        write(f"bank> {result['messages'][-1].text}\n")


@contextlib.contextmanager
def background_api() -> Iterator[str]:
    """Run the bank API on a free local port in a background thread; yield its URL.

    MCP servers are separate processes, so they need a real HTTP endpoint, not an in-process app.
    """
    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config("securebank.api:app", host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("bank API failed to start")
        threading.Event().wait(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


async def start_chat(api_url: str, customer_id: str, settings: Settings) -> int:
    from securebank.agents import build_llm, build_supervisor, open_mcp_toolsets, stdio_connections

    try:
        client = BankClient.login(httpx.Client(base_url=api_url, timeout=30), customer_id)
    except BankAPIError as e:
        print(f"Login failed: {e.detail}", file=sys.stderr)
        return 1
    except httpx.ConnectError:
        print(f"Couldn't reach the bank API at {api_url}.", file=sys.stderr)
        return 1

    async with contextlib.AsyncExitStack() as stack:
        toolsets = await open_mcp_toolsets(stack, stdio_connections(api_url, client.token))
        graph = build_supervisor(toolsets, lambda effort: build_llm(settings, effort))
        await run_chat(graph, client)
    return 0


def chat(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).", file=sys.stderr)
        return 1
    if args.api_url:
        return asyncio.run(start_chat(args.api_url, args.customer, settings))
    with background_api() as api_url:
        return asyncio.run(start_chat(api_url, args.customer, settings))


def serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run("securebank.api:app", host=args.host, port=args.port)
    return 0


def mcp(args: argparse.Namespace, settings: Settings) -> int:
    from securebank import mcp_server

    argv = ["--role", args.role, "--api-url", args.api_url]
    if args.customer:
        argv += ["--customer", args.customer]
    return mcp_server.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="securebank", description="SecureBank dispute assistant")
    sub = parser.add_subparsers(dest="command")

    chat_p = sub.add_parser("chat", help="chat with the assistant as a customer")
    chat_p.add_argument("--customer", default="CUST-001", help="customer to sign in as")
    chat_p.add_argument("--api-url", help="use a running API server instead of an in-process one")
    chat_p.set_defaults(func=chat)

    serve_p = sub.add_parser("serve", help="run the bank API server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.set_defaults(func=serve)

    mcp_p = sub.add_parser("mcp", help="run an MCP server over stdio (e.g. for Claude Desktop)")
    mcp_p.add_argument("--role", default="all", help="account, fraud, policy, action, or all")
    mcp_p.add_argument("--api-url", default="http://127.0.0.1:8000")
    mcp_p.add_argument("--customer", help="demo login as this customer")
    mcp_p.set_defaults(func=mcp)

    args = parser.parse_args(argv)
    settings = get_settings()
    if not args.command:
        key_status = "set" if settings.anthropic_api_key else "MISSING"
        print(
            f"SecureBank v{__version__} | model={settings.model} | ANTHROPIC_API_KEY {key_status}"
        )
        parser.print_help()
        return 0
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
