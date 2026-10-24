"""Card dispute rules. The API enforces them on filing and serves them to the policy agent."""

from datetime import date

from securebank.models import (
    Dispute,
    DisputeReason,
    DisputeRule,
    DisputeStatus,
    Eligibility,
    Transaction,
    TransactionStatus,
)

ACTIVE_DISPUTE_STATUSES = {DisputeStatus.OPEN, DisputeStatus.UNDER_REVIEW}

DISPUTE_RULES: dict[DisputeReason, DisputeRule] = {
    rule.reason: rule
    for rule in [
        DisputeRule(
            reason=DisputeReason.FRAUD,
            window_days=120,
            summary="Charges the customer did not authorize. Block the card first so no further "
            "charges go through, then dispute each unauthorized transaction.",
        ),
        DisputeRule(
            reason=DisputeReason.DUPLICATE,
            window_days=60,
            summary="A purchase billed more than once. Dispute the extra copy, not the original.",
        ),
        DisputeRule(
            reason=DisputeReason.NOT_RECEIVED,
            window_days=120,
            summary="Goods or services paid for but never delivered. The customer should contact "
            "the merchant first where possible.",
        ),
        DisputeRule(
            reason=DisputeReason.INCORRECT_AMOUNT,
            window_days=60,
            summary="The charge differs from the amount the customer agreed to pay.",
        ),
    ]
}


def check_eligibility(txn: Transaction, reason: DisputeReason, today: date) -> Eligibility:
    """Check the transaction itself. Existing disputes are checked separately by active_dispute."""
    problems = []
    if txn.status == TransactionStatus.PENDING:
        problems.append("Pending transactions can't be disputed until they post.")
    else:
        rule = DISPUTE_RULES[reason]
        age = (today - txn.posted_on).days
        if age > rule.window_days:
            problems.append(
                f"Posted {age} days ago; {reason} disputes must be filed within "
                f"{rule.window_days} days."
            )
    return Eligibility(eligible=not problems, problems=problems)


def active_dispute(disputes: list[Dispute], transaction_id: str) -> Dispute | None:
    for d in disputes:
        if d.transaction_id == transaction_id and d.status in ACTIVE_DISPUTE_STATUSES:
            return d
    return None
