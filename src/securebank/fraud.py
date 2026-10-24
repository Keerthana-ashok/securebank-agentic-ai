"""Deterministic fraud signals, served by the API. They are evidence for the fraud agent, not a
verdict."""

from collections import defaultdict
from statistics import median

from securebank.models import FraudSignal, SignalKind, Transaction, TransactionStatus

LARGE_AMOUNT_MULTIPLIER = 5
MIN_HISTORY_FOR_AMOUNT_CHECK = 3


def fraud_signals(txns: list[Transaction], home_country: str = "US") -> list[FraudSignal]:
    posted = [t for t in txns if t.status == TransactionStatus.POSTED]
    signals: list[FraudSignal] = []

    groups: dict[tuple, list[Transaction]] = defaultdict(list)
    for t in posted:
        groups[(t.merchant, t.amount, t.posted_on)].append(t)
    for group in groups.values():
        original, *copies = sorted(group, key=lambda t: t.transaction_id)
        for t in copies:
            signals.append(
                FraudSignal(
                    transaction_id=t.transaction_id,
                    kind=SignalKind.POSSIBLE_DUPLICATE,
                    detail=f"Same merchant, amount, and date as {original.transaction_id}.",
                )
            )

    typical = median(t.amount for t in posted) if posted else None
    for t in posted:
        if t.country != home_country:
            signals.append(
                FraudSignal(
                    transaction_id=t.transaction_id,
                    kind=SignalKind.FOREIGN,
                    detail=f"Merchant country {t.country}, outside {home_country}.",
                )
            )
        if (
            typical
            and len(posted) >= MIN_HISTORY_FOR_AMOUNT_CHECK
            and t.amount > LARGE_AMOUNT_MULTIPLIER * typical
        ):
            signals.append(
                FraudSignal(
                    transaction_id=t.transaction_id,
                    kind=SignalKind.UNUSUALLY_LARGE,
                    detail=f"{t.amount} is over {LARGE_AMOUNT_MULTIPLIER}x the card's "
                    f"median charge of {typical}.",
                )
            )
    return signals
