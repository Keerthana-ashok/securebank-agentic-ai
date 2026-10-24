"""HTTP client for the SecureBank API, bound to one customer's session."""

import json

import httpx
from pydantic import TypeAdapter

from securebank.models import (
    Card,
    Customer,
    Dispute,
    DisputeReason,
    DisputeRule,
    Eligibility,
    FraudSignal,
    Transaction,
)

_cards = TypeAdapter(list[Card])
_transactions = TypeAdapter(list[Transaction])
_disputes = TypeAdapter(list[Dispute])
_rules = TypeAdapter(list[DisputeRule])
_signals = TypeAdapter(list[FraudSignal])


class BankAPIError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class BankClient:
    """Every request carries the session token, so the API scopes results to this customer."""

    def __init__(self, http: httpx.Client, customer_id: str, token: str) -> None:
        self._http = http
        self.customer_id = customer_id
        self.token = token

    @classmethod
    def login(cls, http: httpx.Client, customer_id: str) -> "BankClient":
        resp = http.post("/auth/session", json={"customer_id": customer_id})
        _raise_for_error(resp)
        return cls.from_token(http, resp.json()["token"])

    @classmethod
    def from_token(cls, http: httpx.Client, token: str) -> "BankClient":
        """Join an existing session, e.g. one handed to an MCP server by its parent process."""
        http.headers["Authorization"] = f"Bearer {token}"
        resp = http.get("/me")
        _raise_for_error(resp)
        return cls(http, resp.json()["customer_id"], token)

    def me(self) -> Customer:
        return Customer.model_validate(self._request("GET", "/me"))

    def list_cards(self) -> list[Card]:
        return _cards.validate_python(self._request("GET", "/cards"))

    def list_transactions(self, card_id: str) -> list[Transaction]:
        return _transactions.validate_python(self._request("GET", f"/cards/{card_id}/transactions"))

    def get_transaction(self, transaction_id: str) -> Transaction:
        return Transaction.model_validate(self._request("GET", f"/transactions/{transaction_id}"))

    def list_disputes(self) -> list[Dispute]:
        return _disputes.validate_python(self._request("GET", "/disputes"))

    def get_dispute(self, dispute_id: str) -> Dispute:
        return Dispute.model_validate(self._request("GET", f"/disputes/{dispute_id}"))

    def fraud_signals(self, card_id: str) -> list[FraudSignal]:
        return _signals.validate_python(self._request("GET", f"/cards/{card_id}/fraud-signals"))

    def dispute_policy(self, reason: DisputeReason | None = None) -> list[DisputeRule]:
        params = {"reason": reason} if reason else None
        return _rules.validate_python(self._request("GET", "/policy/disputes", params=params))

    def dispute_eligibility(self, transaction_id: str, reason: DisputeReason) -> Eligibility:
        path = f"/transactions/{transaction_id}/eligibility"
        return Eligibility.model_validate(self._request("GET", path, params={"reason": reason}))

    def block_card(self, card_id: str, reason: str | None = None) -> Card:
        body = {"reason": reason}
        return Card.model_validate(self._request("POST", f"/cards/{card_id}/block", json=body))

    def file_dispute(self, transaction_id: str, reason: DisputeReason, description: str) -> Dispute:
        body = {"transaction_id": transaction_id, "reason": reason, "description": description}
        return Dispute.model_validate(self._request("POST", "/disputes", json=body))

    def _request(self, method: str, path: str, **kwargs):
        resp = self._http.request(method, path, **kwargs)
        _raise_for_error(resp)
        return resp.json()


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    if not isinstance(detail, str):
        detail = json.dumps(detail)  # request validation errors arrive as a list
    raise BankAPIError(resp.status_code, detail)
