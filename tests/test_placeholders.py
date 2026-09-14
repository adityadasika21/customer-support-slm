"""Tests for the placeholder ablation.

The substitution rule carries a safety property worth testing explicitly: entities the model
cannot know (support phone numbers, portal URLs) must *never* be replaced with invented values.
Baking fabricated contact details into a support model's weights is a real harm, because a
customer may act on them.
"""

from __future__ import annotations

from csbot.data.placeholders import (
    DERIVABLE_GENERATORS,
    find_placeholders,
    is_derivable,
    raw_example,
    substitute_example,
)


def test_find_placeholders():
    found = find_placeholders("cancel order {{Order Number}} for {{Person Name}}")
    assert found == {"Order Number", "Person Name"}


def test_find_placeholders_tolerates_whitespace():
    assert find_placeholders("{{ Order Number }}") == {"Order Number"}


def test_raw_variant_is_untouched():
    """Variant A must be byte-identical to the source."""
    instruction = "cancel order {{Order Number}}"
    response = "I will cancel {{Order Number}} right away."
    result = raw_example(instruction, response)
    assert result.instruction == instruction
    assert result.response == response
    assert result.substituted == {}


def test_derivable_entity_substituted_consistently():
    """The same generated value must appear on both sides, or the pair teaches nothing."""
    result = substitute_example(
        "I need to cancel order {{Order Number}}",
        "Certainly, I am cancelling order {{Order Number}} now.",
        seed=1,
    )
    assert "{{Order Number}}" not in result.instruction
    assert "{{Order Number}}" not in result.response

    value = result.substituted["Order Number"]
    assert value in result.instruction
    assert value in result.response, "instruction and response must carry the same value"


def test_non_derivable_placeholder_is_retained():
    """A support phone number is not knowable from context and must stay a placeholder.

    Substituting it would train the model to state a fabricated phone number as fact.
    """
    result = substitute_example(
        "how do I contact you",
        "Call us on {{Customer Support Phone Number}}.",
        seed=1,
    )
    assert "{{Customer Support Phone Number}}" in result.response
    assert "Customer Support Phone Number" in result.retained
    assert result.substituted == {}


def test_response_only_placeholder_is_retained():
    """A placeholder absent from the customer's message was never supplied, so it is not
    derivable from context even when its type normally would be."""
    result = substitute_example(
        "I want to cancel my order",
        "I will cancel order {{Order Number}} for you.",
        seed=1,
    )
    assert "{{Order Number}}" in result.response
    assert result.substituted == {}


def test_substitution_is_deterministic_for_a_seed():
    """A rebuild must reproduce byte-identical training data."""
    args = ("cancel order {{Order Number}}", "Cancelling {{Order Number}}.")
    assert substitute_example(*args, seed=7) == substitute_example(*args, seed=7)


def test_substitution_varies_across_seeds():
    """Fixed values would let the model memorise one constant instead of learning to copy."""
    values = {
        substitute_example(
            "cancel order {{Order Number}}", "Cancelling {{Order Number}}.", seed=s
        ).substituted["Order Number"]
        for s in range(40)
    }
    assert len(values) > 5, "generated order numbers are not varied enough"


def test_order_number_formats_vary():
    """Several surface formats, so 'copy the entity' cannot collapse into 'emit this pattern'."""
    from random import Random

    values = [DERIVABLE_GENERATORS["order number"](Random(s)) for s in range(60)]
    assert any(v.startswith("#") for v in values)
    assert any(v.startswith("ORD-") for v in values)
    assert any(v.isdigit() for v in values)


def test_is_derivable_classification():
    assert is_derivable("Order Number")
    assert is_derivable("order number")  # case-insensitive
    assert not is_derivable("Customer Support Phone Number")
    assert not is_derivable("Website URL")


def test_unknown_placeholder_left_alone():
    """An unrecognised placeholder is treated as non-derivable -- fail safe, not fail open."""
    result = substitute_example(
        "check {{Some New Field}}", "Checking {{Some New Field}}.", seed=1
    )
    assert "{{Some New Field}}" in result.instruction
    assert "{{Some New Field}}" in result.response
