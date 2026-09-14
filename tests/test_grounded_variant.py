"""The grounded variant, and the invariant whose absence cost 30.5% of responses.

Variant B kept non-context-derivable slots as ``{{...}}`` so the model would not learn to
fabricate contact details. It learned to print the slot instead: on the held-out set the
fine-tuned model showed a customer a literal template slot in 30.5% of replies, against 0.1% for
the base model. Fourteen automatic metrics missed it; a human labelling twenty blind pairs found
it. These tests exist so that cannot recur silently.
"""

from __future__ import annotations

import re

import pytest

from csbot.data.placeholders import (
    BUSINESS_FACTS,
    CUSTOMER_DATA,
    UI_ELEMENTS,
    generic_phrase,
    grounded_example,
    substitute_example,
)
from csbot.eval.metrics import hygiene

BRACES = re.compile(r"\{\{|\}\}")


def test_no_brace_survives_a_mapped_response():
    """The invariant. A shipped training row must never contain a template slot."""
    r = grounded_example(
        "I need the support number",
        "Call {{Customer Support Phone Number}} during {{Customer Support Hours}}, or visit "
        "{{Website URL}} and open the '{{Settings}}' section with {{Invoice Number}} to hand.",
        seed=1,
    )
    assert not BRACES.search(r.response), r.response
    assert r.retained == []


def test_unmapped_slots_are_reported_not_silently_kept():
    """An unknown slot must surface in `retained` so the builder can drop the row.

    Silently leaving it is precisely the failure mode being fixed: the text looks fine to a
    length or fluency check and is broken to a customer.
    """
    r = grounded_example("hi", "See {{Some Slot We Never Mapped}} for details.", seed=1)
    assert r.retained == ["Some Slot We Never Mapped"]
    assert "{{" in r.response  # kept verbatim so the row is recognisably unusable


def test_business_facts_are_never_replaced_with_a_concrete_value():
    """Honest generic text, never an invented phone number, URL or opening hours.

    Fabricating these would bake a false fact into the weights that a customer might act on --
    the original and still-valid reason variant B refused to substitute them.
    """
    for phrase in BUSINESS_FACTS.values():
        assert not re.search(r"\d{3,}", phrase), phrase           # no phone-number-ish digits
        assert not re.search(r"https?://|www\.", phrase), phrase  # no invented URL
        assert not re.search(r"\d\s*(am|pm)", phrase, re.I), phrase  # no invented hours


def test_ui_elements_keep_their_words():
    """`the '{{Settings}}' section` is correct once the braces go; do not paraphrase it away."""
    assert generic_phrase("Settings") == "Settings"
    assert generic_phrase("Order Status") == "Order Status"


def test_customer_data_becomes_a_prompt_to_ask():
    assert generic_phrase("Invoice Number") == "your invoice number"
    assert generic_phrase("Tracking Number") == "your tracking number"


def test_sentence_initial_slots_are_capitalised():
    r = grounded_example("hi", "{{Website URL}} has the details.", seed=1)
    assert r.response.startswith("Our website"), r.response


def test_grounded_preserves_the_substituted_arm_behaviour():
    """Context-derivable entities must still be copied from the customer's message."""
    inst, resp = "cancel order {{Order Number}}", "Cancelling {{Order Number}} now."
    b = substitute_example(inst, resp, seed=7)
    c = grounded_example(inst, resp, seed=7)
    assert c.response == b.response
    assert not BRACES.search(c.response)


@pytest.mark.parametrize("text,leak", [
    ("Call us on {{Customer Support Phone Number}}.", True),
    ("Call the number listed in your account area.", False),
    ("Use {{ Website URL }} today.", True),
    ("The set {a, b} is fine.", False),
])
def test_placeholder_leak_metric(text, leak):
    h = hygiene(text)
    assert h.placeholder_leak is leak
    if leak:
        assert not h.clean, "a leaked slot must fail hygiene, not just be recorded"


def test_every_mapping_table_is_lowercase_keyed():
    """generic_phrase() lowercases before lookup; a capitalised key would never match."""
    for table in (BUSINESS_FACTS, CUSTOMER_DATA):
        for k in table:
            assert k == k.lower(), k
    for k in UI_ELEMENTS:
        assert k == k.lower(), k
