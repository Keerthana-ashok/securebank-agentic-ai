from datetime import date

import pytest

from securebank.data import NotFoundError, seed_store
from securebank.models import CardStatus, DisputeReason, DisputeStatus

TODAY = date(2026, 10, 22)


@pytest.fixture
def store():
    return seed_store(TODAY)


def test_cards_are_scoped_to_customer(store):
    assert {c.card_id for c in store.list_cards("CUST-001")} == {"CARD-1001", "CARD-1002"}
    assert {c.card_id for c in store.list_cards("CUST-002")} == {"CARD-2001"}


def test_transactions_sorted_newest_first(store):
    txns = store.list_transactions("CARD-1001")
    assert [t.posted_on for t in txns] == sorted((t.posted_on for t in txns), reverse=True)
    assert all(t.card_id == "CARD-1001" for t in txns)


def test_seed_dates_are_relative_to_today(store):
    assert (TODAY - store.get_transaction("TXN-0006").posted_on).days == 150


def test_seeded_dispute_exists(store):
    [dispute] = store.list_disputes("CUST-001")
    assert dispute.transaction_id == "TXN-0008"
    assert dispute.status == DisputeStatus.UNDER_REVIEW
    assert store.list_disputes("CUST-002") == []


def test_create_dispute_derives_owner_from_transaction(store):
    dispute = store.create_dispute("TXN-0004", DisputeReason.DUPLICATE, "Charged twice.")
    assert dispute.customer_id == "CUST-001"
    assert dispute.card_id == "CARD-1001"
    assert dispute.dispute_id == "DSP-0002"
    assert store.get_dispute(dispute.dispute_id) is dispute


def test_set_card_status(store):
    store.set_card_status("CARD-1001", CardStatus.BLOCKED)
    assert store.get_card("CARD-1001").status == CardStatus.BLOCKED


def test_missing_records_raise_not_found(store):
    with pytest.raises(NotFoundError):
        store.get_card("CARD-9999")
    with pytest.raises(NotFoundError):
        store.create_dispute("TXN-9999", DisputeReason.FRAUD, "x")
