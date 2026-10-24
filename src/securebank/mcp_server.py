"""SecureBank MCP server: serves one role's bank tools to an MCP client over stdio.

The server is just another client of the bank API. It acts for one customer session, taken
from SECUREBANK_SESSION_TOKEN (or a demo login with --customer), and registers only the tools
for its --role, so a fraud agent's server cannot block cards no matter what the model asks.

    python -m securebank.mcp_server --role fraud --api-url http://127.0.0.1:8000
"""

import argparse
import os
import sys

import httpx
from mcp.server.fastmcp import FastMCP

from securebank.client import BankAPIError, BankClient
from securebank.tools import ROLES, TOOLS_BY_ROLE, build_tools

ALL = "all"  # every role's tools, for clients like Claude Desktop with their own approvals
TOKEN_ENV = "SECUREBANK_SESSION_TOKEN"


def create_mcp_server(client: BankClient, role: str) -> FastMCP:
    if role == ALL:
        names = sorted({name for r in ROLES for name in TOOLS_BY_ROLE[r]})
    else:
        names = TOOLS_BY_ROLE[role]
    server = FastMCP(
        f"securebank-{role}",
        instructions=f"SecureBank card dispute tools ({role}) for customer {client.customer_id}.",
    )
    tools = build_tools(client)
    for name in names:
        server.add_tool(tools[name], name=name)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="securebank-mcp", description=__doc__.split("\n")[0])
    parser.add_argument("--role", choices=[*ROLES, ALL], required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--customer", help=f"demo login as this customer when {TOKEN_ENV} is not set"
    )
    args = parser.parse_args(argv)

    # stdout carries the MCP protocol; diagnostics go to stderr.
    http = httpx.Client(base_url=args.api_url, timeout=30)
    try:
        if token := os.environ.get(TOKEN_ENV):
            client = BankClient.from_token(http, token)
        elif args.customer:
            client = BankClient.login(http, args.customer)
        else:
            print(f"Set {TOKEN_ENV} or pass --customer.", file=sys.stderr)
            return 1
    except BankAPIError as e:
        print(f"Bank API rejected the session: {e.detail}", file=sys.stderr)
        return 1
    except httpx.ConnectError:
        print(f"Couldn't reach the bank API at {args.api_url}.", file=sys.stderr)
        return 1

    create_mcp_server(client, args.role).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
