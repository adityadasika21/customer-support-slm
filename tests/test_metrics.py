"""Tests for the deterministic metrics.

These metrics carry the base-vs-tuned argument, so they are tested on hand-built fixtures where
the right answer is unambiguous. A metric that is subtly wrong is worse than no metric: it
produces confident numbers pointing the wrong way.
"""

from __future__ import annotations

import numpy as np
import pytest

from csbot.eval.metrics import (
    bootstrap_ci,
    entity_fidelity,
    extract_entities,
    hygiene,
    normalize_entity,
    paired_bootstrap,
    repetition_rate,
    win_tie_loss,
)


# -- entity handling -------------------------------------------------------------------------

def test_extract_entities_finds_identifiers():
    assert "12345" in extract_entities("please cancel order 12345")
    assert "#A-7781" in extract_entities("order #A-7781 is late")


def test_extract_entities_finds_placeholders():
    assert "{{Order Number}}" in extract_entities("cancel {{Order Number}}")


def test_prose_numbers_are_not_entities():
    """'3-5 business days' must not count as an identifier, or every polite reply would look
    like it hallucinated an order number."""
    for text in ["within 3-5 business days", "in 24 hours", "about 2 weeks", "10 percent off"]:
        assert extract_entities(text) == set(), f"false positive in {text!r}"


def test_normalize_entity_ignores_surface_form():
    assert normalize_entity("#12345") == normalize_entity("12345")


def test_entity_recall_rewards_echoing():
    score = entity_fidelity("cancel order 12345", "I am cancelling order 12345 now.")
    assert score.recall == 1.0
    assert score.n_hallucinated == 0
    assert score.clean


def test_entity_recall_penalises_dropping():
    score = entity_fidelity("cancel order 12345", "I am cancelling your order now.")
    assert score.recall == 0.0
    assert score.clean, "dropping an id is a recall miss, not a hallucination"


def test_hallucinated_identifier_detected():
    """The dangerous failure: quoting an order number the customer never gave."""
    score = entity_fidelity("I want to cancel my order", "I've cancelled order 99887 for you.")
    assert score.n_hallucinated == 1
    assert not score.clean
    assert "99887" in score.hallucinated[0]


def test_entity_recall_is_nan_without_entities():
    """No entities means recall is undefined, not zero -- averaging in a zero would punish
    every reply to a question that contained no identifier."""
    assert np.isnan(entity_fidelity("how do I get a refund", "Here is how.").recall)


def test_surface_variation_counts_as_echo():
    score = entity_fidelity("order #12345 please", "Order 12345 is on its way.")
    assert score.recall == 1.0
    assert score.n_hallucinated == 0


# -- hygiene ---------------------------------------------------------------------------------

def test_think_leakage_detected():
    assert hygiene("<think>the user wants a refund</think> Sure!").think_leakage
    assert not hygiene("Sure, I can help with that.").think_leakage


def test_refusal_and_boilerplate_detected():
    assert hygiene("As an AI language model, I cannot help.").refusal
    assert hygiene("As an AI language model, I cannot help.").boilerplate


def test_truncation_detected():
    assert hygiene("I will help you with your order and then we can").truncated
    assert not hygiene("I will help you with your order.").truncated


def test_truncation_allows_closing_punctuation():
    assert not hygiene('He said "we can help you."').truncated


def test_repetition_rate():
    assert repetition_rate("a b c d e f g h") == 0.0
    assert repetition_rate(("the same words over " * 6).strip()) > 0.5


def test_repetition_rate_short_text_is_safe():
    assert repetition_rate("hi") == 0.0


def test_clean_reply_is_clean():
    score = hygiene("Thanks for reaching out. I've cancelled order 12345 for you.")
    assert score.clean


def test_empty_reply_is_not_clean():
    assert not hygiene("").clean
    assert hygiene("").empty


# -- aggregation -----------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_mean():
    values = [1.0] * 70 + [0.0] * 30
    ci = bootstrap_ci(values, n_boot=2000, seed=0)
    assert ci.lo < ci.mean < ci.hi
    assert ci.mean == pytest.approx(0.70, abs=1e-9)
    assert ci.n == 100


def test_bootstrap_ci_ignores_nan():
    ci = bootstrap_ci([1.0, float("nan"), 0.0], n_boot=500, seed=0)
    assert ci.n == 2


def test_bootstrap_ci_handles_empty():
    assert bootstrap_ci([], n_boot=100).n == 0


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.1, 300)
    tuned = base + 0.20  # a large, consistent, paired improvement
    diff = paired_bootstrap(base, tuned, n_boot=2000, seed=0)
    assert diff.diff == pytest.approx(0.20, abs=0.01)
    assert diff.significant
    assert diff.lo > 0


def test_paired_bootstrap_reports_no_difference_when_none_exists():
    """The important negative case: identical arms must not look significant."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.5, 0.1, 300)
    tuned = base.copy()
    diff = paired_bootstrap(base, tuned, n_boot=2000, seed=0)
    assert diff.diff == pytest.approx(0.0, abs=1e-9)
    assert not diff.significant


def test_paired_bootstrap_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_win_tie_loss_counts():
    counts = win_tie_loss([0, 0, 1, 1], [1, 1, 1, 0])
    assert counts == {"win": 2, "tie": 1, "loss": 1, "n": 4}


# -- refusal vs empathy ----------------------------------------------------------------------

def test_empathetic_apology_is_not_a_refusal():
    """The metric bug that mattered: in support, apology is the desired register.

    An earlier pattern matched a bare "I'm sorry" and reported a 59% refusal rate for a model
    whose replies were well-formed, empathetic support answers -- penalising exactly the tone we
    are training for.
    """
    empathetic = [
        "I'm sorry to hear your order hasn't arrived. Let me look into that for you.",
        "I am sorry for the inconvenience this has caused. Here's what happens next.",
        "I'm sorry about the delay! I've escalated this for you.",
        "I apologize for the trouble. Your refund is being processed.",
        "I'm afraid the item is out of stock, but I can offer an alternative.",
    ]
    for text in empathetic:
        assert not hygiene(text).refusal, f"false positive refusal: {text!r}"


def test_real_refusals_are_detected():
    refusals = [
        "I'm sorry, but I cannot help with that request.",
        "As an AI language model, I don't have opinions.",
        "I can't provide account details.",
        "Unfortunately, I am unable to process that.",
        "I don't have access to your account information.",
        "I'm not able to assist with this.",
    ]
    for text in refusals:
        assert hygiene(text).refusal, f"missed refusal: {text!r}"


def test_good_support_reply_is_clean():
    text = (
        "I'm sorry to hear that order 12345 hasn't arrived yet. I've checked and it's "
        "currently with the courier. You should receive it within 2 business days."
    )
    score = hygiene(text)
    assert score.clean
    assert not score.refusal


def test_placeholder_in_response_is_not_hallucination():
    """A template slot is a "fill this in" marker, not an invented fact.

    Counting placeholders as hallucinations inverted this metric: gold reference responses scored
    0.674 clean while an untuned base model scored 0.957, purely because the base never emits
    placeholders. The metric rewarded the model that had learned less.
    """
    score = entity_fidelity(
        "how do I contact you about order 12345",
        "Call us on {{Customer Support Phone Number}} about order 12345, or visit {{Website URL}}.",
    )
    assert score.n_hallucinated == 0
    assert score.clean
    assert score.recall == 1.0


def test_concrete_invented_identifier_still_counts():
    """The dangerous case must still fire: a real number the customer never gave."""
    score = entity_fidelity(
        "how do I contact you",
        "Call us on {{Customer Support Phone Number}} about your order 55512.",
    )
    assert score.n_hallucinated == 1
    assert not score.clean
    assert "55512" in score.hallucinated[0]


def test_is_placeholder():
    from csbot.eval.metrics import is_placeholder

    assert is_placeholder("{{Order Number}}")
    assert is_placeholder("  {{Website URL}}  ")
    assert not is_placeholder("12345")
    assert not is_placeholder("#A-7781")


def test_identifier_at_end_of_sentence_is_found():
    """Regression: an earlier lookahead excluded identifiers followed by a period.

    That is where hallucinated order numbers usually appear, so the dangerous case was silently
    invisible.
    """
    assert "55512" in extract_entities("I've cancelled your order 55512.")
    assert "88213" in extract_entities("Your order is 88213!")
    assert "#A-7781" in extract_entities("That would be order #A-7781.")


def test_decimals_are_not_identifiers():
    """Prices and versions must not register as order numbers."""
    for text in ["the total is $49.99", "version 12.04 is current", "rated 4.5 out of 5"]:
        assert extract_entities(text) == set(), f"false positive in {text!r}: {extract_entities(text)}"
