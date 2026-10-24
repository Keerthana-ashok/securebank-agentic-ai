"""In-memory bank data store seeded with mock data.

This layer performs no authorization. Access control is enforced one level up, in the
FastAPI backend (securebank.api), so that agents never touch the store directly.
"""

import secrets
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import count

from securebank.models import (
    Card,
    CardStatus,
    CardType,
    Customer,
    Dispute,
    DisputeReason,
    DisputeStatus,
    Transaction,
    TransactionStatus,
)


class NotFoundError(LookupError):
    pass


class BankStore:
    def __init__(self, today: date | None = None) -> None:
        self.today = today or date.today()
        self.customers: dict[str, Customer] = {}
        self.cards: dict[str, Card] = {}
        self.transactions: dict[str, Transaction] = {}
        self.disputes: dict[str, Dispute] = {}
        self.sessions: dict[str, str] = {}  # token -> customer_id
        self._dispute_seq = count(1)

    # Sessions

    def create_session(self, customer_id: str) -> str:
        self.get_customer(customer_id)
        token = secrets.token_urlsafe(32)
        self.sessions[token] = customer_id
        return token

    def resolve_session(self, token: str) -> str:
        return self._get(self.sessions, token, "session")

    # Reads

    def get_customer(self, customer_id: str) -> Customer:
        return self._get(self.customers, customer_id, "customer")

    def get_card(self, card_id: str) -> Card:
        return self._get(self.cards, card_id, "card")

    def get_transaction(self, transaction_id: str) -> Transaction:
        return self._get(self.transactions, transaction_id, "transaction")

    def get_dispute(self, dispute_id: str) -> Dispute:
        return self._get(self.disputes, dispute_id, "dispute")

    def list_cards(self, customer_id: str) -> list[Card]:
        return [c for c in self.cards.values() if c.customer_id == customer_id]

    def list_transactions(self, card_id: str) -> list[Transaction]:
        txns = [t for t in self.transactions.values() if t.card_id == card_id]
        return sorted(txns, key=lambda t: t.posted_on, reverse=True)

    def list_disputes(self, customer_id: str) -> list[Dispute]:
        return [d for d in self.disputes.values() if d.customer_id == customer_id]

    # Writes

    def set_card_status(self, card_id: str, status: CardStatus) -> Card:
        card = self.get_card(card_id)
        card.status = status
        return card

    def create_dispute(
        self, transaction_id: str, reason: DisputeReason, description: str
    ) -> Dispute:
        txn = self.get_transaction(transaction_id)
        card = self.get_card(txn.card_id)
        dispute = Dispute(
            dispute_id=f"DSP-{next(self._dispute_seq):04d}",
            transaction_id=txn.transaction_id,
            card_id=card.card_id,
            customer_id=card.customer_id,
            reason=reason,
            description=description,
            created_at=datetime.now(UTC),
        )
        self.disputes[dispute.dispute_id] = dispute
        return dispute

    @staticmethod
    def _get(table: dict, key: str, kind: str):
        try:
            return table[key]
        except KeyError:
            raise NotFoundError(f"{kind} {key!r} not found") from None


def seed_store(today: date | None = None) -> BankStore:
    """Build a store with two customers and dispute-worthy scenarios.

    Dates are relative to `today` so dispute time windows behave the same on any run.
    """
    store = BankStore(today)
    ago = lambda days: store.today - timedelta(days=days)  # noqa: E731

    for c in [
        Customer(customer_id="CUST-001", name="Alex Rivera", email="alex@example.com"),
        Customer(customer_id="CUST-002", name="Sam Chen", email="sam@example.com"),
    ]:
        store.customers[c.customer_id] = c

    for card in [
        Card(card_id="CARD-1001", customer_id="CUST-001", card_type=CardType.CREDIT, last4="4242"),
        Card(card_id="CARD-1002", customer_id="CUST-001", card_type=CardType.DEBIT, last4="1881"),
        Card(card_id="CARD-2001", customer_id="CUST-002", card_type=CardType.CREDIT, last4="5555"),
    ]:
        store.cards[card.card_id] = card

    txns = [
        # CUST-001 credit card
        ("TXN-0001", "CARD-1001", ago(2), "Blue Bottle Coffee", "6.50", "US"),
        ("TXN-0002", "CARD-1001", ago(5), "StreamFlix Subscription", "15.99", "US"),
        # Duplicate charge: same merchant, amount, and day.
        ("TXN-0003", "CARD-1001", ago(7), "ElectroMart Online", "249.00", "US"),
        ("TXN-0004", "CARD-1001", ago(7), "ElectroMart Online", "249.00", "US"),
        # Suspicious: foreign merchant, large amount, unlike the customer's usual spend.
        ("TXN-0005", "CARD-1001", ago(1), "LUXGOODS VILNIUS", "1899.00", "LT"),
        # Too old for most dispute windows.
        ("TXN-0006", "CARD-1001", ago(150), "Hotel Marina", "412.30", "US"),
        # CUST-001 debit card
        ("TXN-0007", "CARD-1002", ago(3), "Grocery Hub", "84.12", "US"),
        ("TXN-0008", "CARD-1002", ago(12), "QuickShip Marketplace", "59.99", "US"),
        # CUST-002 credit card
        ("TXN-0009", "CARD-2001", ago(4), "City Parking", "18.00", "US"),
        ("TXN-0010", "CARD-2001", ago(9), "FlyAway Airlines", "530.00", "US"),
    ]
    for txn_id, card_id, posted_on, merchant, amount, country in txns:
        store.transactions[txn_id] = Transaction(
            transaction_id=txn_id,
            card_id=card_id,
            posted_on=posted_on,
            merchant=merchant,
            amount=Decimal(amount),
            country=country,
        )
    store.transactions["TXN-0011"] = Transaction(
        transaction_id="TXN-0011",
        card_id="CARD-1001",
        posted_on=store.today,
        merchant="Gas & Go",
        amount=Decimal("45.00"),
        status=TransactionStatus.PENDING,
    )

    # An existing dispute, so "what's the status of my dispute" has an answer.
    existing = store.create_dispute(
        "TXN-0008", DisputeReason.NOT_RECEIVED, "Order never arrived after 10 days."
    )
    existing.status = DisputeStatus.UNDER_REVIEW

    return store
