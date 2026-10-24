"""Domain models for customers, cards, transactions, and disputes."""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class CardType(StrEnum):
    CREDIT = "credit"
    DEBIT = "debit"


class CardStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"


class TransactionStatus(StrEnum):
    PENDING = "pending"
    POSTED = "posted"


class DisputeReason(StrEnum):
    FRAUD = "fraud"  # customer did not authorize the transaction
    DUPLICATE = "duplicate"  # charged more than once for the same purchase
    NOT_RECEIVED = "not_received"  # goods or services never arrived
    INCORRECT_AMOUNT = "incorrect_amount"


class DisputeStatus(StrEnum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    REJECTED = "rejected"


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str


class Card(BaseModel):
    card_id: str
    customer_id: str
    card_type: CardType
    last4: str = Field(pattern=r"^\d{4}$")
    status: CardStatus = CardStatus.ACTIVE


class Transaction(BaseModel):
    transaction_id: str
    card_id: str
    posted_on: date
    merchant: str
    amount: Decimal
    currency: str = "USD"
    country: str = "US"
    status: TransactionStatus = TransactionStatus.POSTED


class Dispute(BaseModel):
    dispute_id: str
    transaction_id: str
    card_id: str
    customer_id: str
    reason: DisputeReason
    description: str
    status: DisputeStatus = DisputeStatus.OPEN
    created_at: datetime


class DisputeRule(BaseModel):
    reason: DisputeReason
    window_days: int
    summary: str


class Eligibility(BaseModel):
    eligible: bool
    problems: list[str] = []


class SignalKind(StrEnum):
    POSSIBLE_DUPLICATE = "possible_duplicate"
    FOREIGN = "foreign"
    UNUSUALLY_LARGE = "unusually_large"


class FraudSignal(BaseModel):
    transaction_id: str
    kind: SignalKind
    detail: str
