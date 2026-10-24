"""Bank tools and the least-privilege role each agent gets. Served to agents by the MCP server.

Tools close over a customer-bound BankClient, so the model never supplies a customer ID or
token and cannot reach another customer's data. Only the action role gets tools that change
anything. The API audits every action, whoever calls it.
"""

import functools
import json
from collections.abc import Callable

from pydantic import BaseModel

from securebank.client import BankAPIError, BankClient
from securebank.models import DisputeReason

ACCOUNT = "account"
FRAUD = "fraud"
POLICY = "policy"
ACTION = "action"
ROLES = (ACCOUNT, FRAUD, POLICY, ACTION)

TOOLS_BY_ROLE: dict[str, tuple[str, ...]] = {
    ACCOUNT: (
        "get_profile",
        "list_cards",
        "list_transactions",
        "get_transaction",
        "list_disputes",
        "get_dispute",
    ),
    FRAUD: ("list_cards", "list_transactions", "get_transaction", "find_fraud_signals"),
    POLICY: ("get_dispute_policy", "check_dispute_eligibility"),
    ACTION: ("block_card", "file_dispute"),
}
WRITE_TOOLS = frozenset(TOOLS_BY_ROLE[ACTION])


def _dump(value) -> str:
    if isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
    elif isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value)


def _api_errors_as_text(fn):
    """Return API errors to the model as tool output so it can recover or explain."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BankAPIError as e:
            return f"Error ({e.status_code}): {e.detail}"

    return wrapper


def build_tools(client: BankClient) -> dict[str, Callable[..., str]]:
    """All tool functions, by name. Name, signature, and docstring define each tool's schema."""

    # Reads

    def get_profile() -> str:
        """Get the signed-in customer's name and email."""
        return _dump(client.me())

    def list_cards() -> str:
        """List the customer's cards with type, last 4 digits, and status."""
        return _dump(client.list_cards())

    def list_transactions(card_id: str) -> str:
        """List transactions on one of the customer's cards, newest first."""
        return _dump(client.list_transactions(card_id))

    def get_transaction(transaction_id: str) -> str:
        """Get one transaction by ID."""
        return _dump(client.get_transaction(transaction_id))

    def list_disputes() -> str:
        """List the customer's disputes and their statuses."""
        return _dump(client.list_disputes())

    def get_dispute(dispute_id: str) -> str:
        """Get one dispute by ID."""
        return _dump(client.get_dispute(dispute_id))

    def find_fraud_signals(card_id: str) -> str:
        """Check a card's transactions for possible duplicates, foreign charges, and unusually
        large amounts. Signals are evidence to weigh, not proof of fraud."""
        return _dump(client.fraud_signals(card_id))

    # Policy

    def get_dispute_policy(reason: DisputeReason | None = None) -> str:
        """Get dispute rules and filing windows, for one reason or for all of them."""
        return _dump(client.dispute_policy(reason))

    def check_dispute_eligibility(transaction_id: str, reason: DisputeReason) -> str:
        """Check whether a transaction can be disputed for the given reason today."""
        return _dump(client.dispute_eligibility(transaction_id, reason))

    # Actions

    def block_card(card_id: str, reason: str) -> str:
        """Block one of the customer's cards so no new charges go through. Only call this after
        the customer has confirmed. `reason` is a short note for the audit log."""
        return _dump(client.block_card(card_id, reason))

    def file_dispute(transaction_id: str, reason: DisputeReason, description: str) -> str:
        """File a dispute on a transaction. Only call this after the customer has confirmed.
        `description` is the customer's account of what happened."""
        return _dump(client.file_dispute(transaction_id, reason, description))

    fns = [
        get_profile,
        list_cards,
        list_transactions,
        get_transaction,
        list_disputes,
        get_dispute,
        find_fraud_signals,
        get_dispute_policy,
        check_dispute_eligibility,
        block_card,
        file_dispute,
    ]
    return {fn.__name__: _api_errors_as_text(fn) for fn in fns}
