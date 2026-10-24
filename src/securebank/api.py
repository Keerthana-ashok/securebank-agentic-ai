"""FastAPI backend over the bank store.

Every route except login is scoped to the customer behind the bearer token. Records owned by
another customer return 404, the same as records that don't exist, so the API never reveals
whether someone else's card or transaction exists.
"""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from securebank import __version__
from securebank.audit import AuditLog
from securebank.config import get_settings
from securebank.data import BankStore, NotFoundError, seed_store
from securebank.fraud import fraud_signals
from securebank.models import (
    Card,
    CardStatus,
    Customer,
    Dispute,
    DisputeReason,
    DisputeRule,
    Eligibility,
    FraudSignal,
    Transaction,
)
from securebank.policy import DISPUTE_RULES, active_dispute, check_eligibility


class LoginRequest(BaseModel):
    customer_id: str


class LoginResponse(BaseModel):
    token: str
    customer_id: str


class DisputeRequest(BaseModel):
    transaction_id: str
    reason: DisputeReason
    description: str = Field(min_length=1, max_length=2000)


bearer = HTTPBearer(auto_error=False)


class BlockRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


def get_store(request: Request) -> BankStore:
    return request.app.state.store


Store = Annotated[BankStore, Depends(get_store)]


def get_audit(request: Request) -> AuditLog:
    return request.app.state.audit


Audit = Annotated[AuditLog, Depends(get_audit)]


def audited(audit: AuditLog, customer_id: str, action: str, args: dict, call):
    """Run an action and record the attempt, including ones the API rejects."""
    try:
        result = call()
    except HTTPException as e:
        audit.record(
            customer_id=customer_id,
            action=action,
            args=args,
            outcome="error",
            status=e.status_code,
            detail=e.detail,
        )
        raise
    audit.record(
        customer_id=customer_id,
        action=action,
        args=args,
        outcome="ok",
        result=result.model_dump(mode="json"),
    )
    return result


def get_customer_id(
    store: Store, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> str:
    try:
        if creds is None:
            raise NotFoundError
        return store.resolve_session(creds.credentials)
    except NotFoundError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid or missing session token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


CustomerId = Annotated[str, Depends(get_customer_id)]


def not_found(kind: str, key: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"{kind} {key!r} not found")


def owned_card(store: BankStore, customer_id: str, card_id: str) -> Card:
    try:
        card = store.get_card(card_id)
    except NotFoundError:
        raise not_found("card", card_id) from None
    if card.customer_id != customer_id:
        raise not_found("card", card_id)
    return card


def owned_transaction(store: BankStore, customer_id: str, transaction_id: str) -> Transaction:
    try:
        txn = store.get_transaction(transaction_id)
        card = store.get_card(txn.card_id)
    except NotFoundError:
        raise not_found("transaction", transaction_id) from None
    if card.customer_id != customer_id:
        raise not_found("transaction", transaction_id)
    return txn


def create_app(store: BankStore | None = None, audit: AuditLog | None = None) -> FastAPI:
    app = FastAPI(title="SecureBank API", version=__version__)
    app.state.store = store or seed_store()
    app.state.audit = audit or AuditLog(get_settings().audit_log_path)

    @app.post("/auth/session", response_model=LoginResponse)
    def login(body: LoginRequest, store: Store) -> LoginResponse:
        # Mock login: stands in for real customer authentication, which is out of scope.
        try:
            token = store.create_session(body.customer_id)
        except NotFoundError:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown customer") from None
        return LoginResponse(token=token, customer_id=body.customer_id)

    @app.get("/me", response_model=Customer)
    def me(store: Store, customer_id: CustomerId) -> Customer:
        return store.get_customer(customer_id)

    @app.get("/cards", response_model=list[Card])
    def list_cards(store: Store, customer_id: CustomerId) -> list[Card]:
        return store.list_cards(customer_id)

    @app.get("/cards/{card_id}", response_model=Card)
    def get_card(card_id: str, store: Store, customer_id: CustomerId) -> Card:
        return owned_card(store, customer_id, card_id)

    @app.get("/cards/{card_id}/transactions", response_model=list[Transaction])
    def list_transactions(card_id: str, store: Store, customer_id: CustomerId) -> list[Transaction]:
        owned_card(store, customer_id, card_id)
        return store.list_transactions(card_id)

    @app.get("/cards/{card_id}/fraud-signals", response_model=list[FraudSignal])
    def get_fraud_signals(card_id: str, store: Store, customer_id: CustomerId) -> list[FraudSignal]:
        owned_card(store, customer_id, card_id)
        return fraud_signals(store.list_transactions(card_id))

    @app.post("/cards/{card_id}/block", response_model=Card)
    def block_card(
        card_id: str,
        store: Store,
        audit: Audit,
        customer_id: CustomerId,
        body: BlockRequest | None = None,
    ) -> Card:
        def block() -> Card:
            owned_card(store, customer_id, card_id)
            return store.set_card_status(card_id, CardStatus.BLOCKED)

        args = {"card_id": card_id, "reason": body.reason if body else None}
        return audited(audit, customer_id, "block_card", args, block)

    @app.get("/transactions/{transaction_id}", response_model=Transaction)
    def get_transaction(transaction_id: str, store: Store, customer_id: CustomerId) -> Transaction:
        return owned_transaction(store, customer_id, transaction_id)

    @app.get("/transactions/{transaction_id}/eligibility", response_model=Eligibility)
    def get_dispute_eligibility(
        transaction_id: str, reason: DisputeReason, store: Store, customer_id: CustomerId
    ) -> Eligibility:
        txn = owned_transaction(store, customer_id, transaction_id)
        eligibility = check_eligibility(txn, reason, store.today)
        if existing := active_dispute(store.list_disputes(customer_id), transaction_id):
            eligibility.eligible = False
            eligibility.problems.append(f"Already has active dispute {existing.dispute_id}.")
        return eligibility

    @app.get("/policy/disputes", response_model=list[DisputeRule])
    def get_dispute_policy(
        customer_id: CustomerId, reason: DisputeReason | None = None
    ) -> list[DisputeRule]:
        return [DISPUTE_RULES[reason]] if reason else list(DISPUTE_RULES.values())

    @app.get("/disputes", response_model=list[Dispute])
    def list_disputes(store: Store, customer_id: CustomerId) -> list[Dispute]:
        return store.list_disputes(customer_id)

    @app.get("/disputes/{dispute_id}", response_model=Dispute)
    def get_dispute(dispute_id: str, store: Store, customer_id: CustomerId) -> Dispute:
        try:
            dispute = store.get_dispute(dispute_id)
        except NotFoundError:
            raise not_found("dispute", dispute_id) from None
        if dispute.customer_id != customer_id:
            raise not_found("dispute", dispute_id)
        return dispute

    @app.post("/disputes", response_model=Dispute, status_code=status.HTTP_201_CREATED)
    def create_dispute(
        body: DisputeRequest, store: Store, audit: Audit, customer_id: CustomerId
    ) -> Dispute:
        def create() -> Dispute:
            txn = owned_transaction(store, customer_id, body.transaction_id)
            if existing := active_dispute(store.list_disputes(customer_id), txn.transaction_id):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"transaction already has active dispute {existing.dispute_id}",
                )
            # Enforced here too, so an agent that skips the policy check still can't file.
            eligibility = check_eligibility(txn, body.reason, store.today)
            if not eligibility.eligible:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, " ".join(eligibility.problems)
                )
            return store.create_dispute(body.transaction_id, body.reason, body.description)

        args = body.model_dump(mode="json")
        return audited(audit, customer_id, "file_dispute", args, create)

    return app


app = create_app()
