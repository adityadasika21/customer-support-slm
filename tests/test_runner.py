"""Tests for the evaluation orchestration.

These exist to de-risk the critical path: ``scripts/evaluate.py`` produces the headline
base-vs-tuned numbers and runs only once, at the end, after hours of training. A bug discovered
there is a bug discovered expensively, so the scoring, pairing and comparison logic is exercised
here with synthetic generations and no GPU.
"""

from __future__ import annotations

import pandas as pd
import pytest

from csbot.eval.runner import (
    HIGHER_IS_BETTER,
    compare,
    failure_cases,
    save_comparison,
    score_generations,
)


def make_rows(n: int = 6) -> pd.DataFrame:
    """Minimal frame with the columns ``score_generations`` requires."""
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "instruction": [f"please cancel order {1000 + i}" for i in range(n)],
            "response": [f"Certainly, order {1000 + i} is cancelled." for i in range(n)],
            "intent": ["cancel_order"] * n,
            "category": ["ORDER"] * n,
            "flags": ["B"] * n,
        }
    )


def test_score_generations_shape_and_columns():
    rows = make_rows()
    gens = [f"I have cancelled order {1000 + i}." for i in range(len(rows))]
    scored = score_generations(rows, gens)

    assert len(scored) == len(rows)
    for col in ("entity_recall", "hygiene_clean", "think_leakage", "rouge_l", "generation"):
        assert col in scored.columns
    # ids must survive in order -- the paired comparison depends on it
    assert scored["id"].tolist() == rows["id"].tolist()


def test_score_generations_rejects_length_mismatch():
    """A silent zip() truncation here would misalign every downstream metric."""
    with pytest.raises(ValueError, match="mismatch"):
        score_generations(make_rows(4), ["a", "b"])


def test_echoing_the_order_number_scores_full_recall():
    rows = make_rows(3)
    gens = [f"Order {1000 + i} is cancelled." for i in range(3)]
    scored = score_generations(rows, gens)
    assert scored["entity_recall"].mean() == 1.0
    assert scored["entity_clean"].mean() == 1.0


def test_inventing_an_order_number_is_flagged():
    rows = make_rows(2)
    gens = ["I cancelled order 999999 for you.", "I cancelled order 888888 for you."]
    scored = score_generations(rows, gens)
    assert scored["entity_clean"].mean() == 0.0
    assert scored["n_hallucinated"].sum() == 2


def test_compare_produces_ci_and_win_tie_loss():
    rows = make_rows(20)
    base = score_generations(rows, ["I cannot help with that." for _ in range(20)])
    tuned = score_generations(
        rows, [f"Certainly, order {1000 + i} is cancelled." for i in range(20)]
    )

    result = compare(base, tuned, slice_name="t", base_name="b", tuned_name="t")

    assert result.n == 20
    assert "hygiene_clean" in result.metrics
    m = result.metrics["hygiene_clean"]
    assert m["tuned"] > m["base"], "clean replies must beat refusals on hygiene"
    assert m["significant"], "a total, consistent difference must register as significant"
    assert m["win_tie_loss"]["win"] + m["win_tie_loss"]["tie"] + m["win_tie_loss"]["loss"] == 20
    assert len(m["ci"]) == 2 and m["ci"][0] <= m["diff"] <= m["ci"][1]


def test_compare_rejects_unpaired_frames():
    """The whole statistical argument rests on both arms answering the same prompts in order."""
    rows = make_rows(5)
    base = score_generations(rows, ["x"] * 5)
    tuned = score_generations(rows.iloc[::-1].reset_index(drop=True), ["x"] * 5)
    with pytest.raises(ValueError, match="identical ids"):
        compare(base, tuned, slice_name="t", base_name="b", tuned_name="t")


def test_identical_arms_are_not_significant():
    """The negative control: the same generations must never look like an improvement."""
    rows = make_rows(20)
    gens = [f"Order {1000 + i} is cancelled." for i in range(20)]
    a = score_generations(rows, gens)
    b = score_generations(rows, gens)
    result = compare(a, b, slice_name="t", base_name="b", tuned_name="t")
    for name, m in result.metrics.items():
        assert not m["significant"], f"{name} claimed significance on identical arms"
        assert m["diff"] == pytest.approx(0.0, abs=1e-9)


def test_markdown_renders_every_metric():
    rows = make_rows(8)
    base = score_generations(rows, ["I cannot assist." for _ in range(8)])
    tuned = score_generations(rows, [f"Order {1000 + i} cancelled." for i in range(8)])
    md = compare(base, tuned, slice_name="test_indomain", base_name="b", tuned_name="t").to_markdown()

    assert "test_indomain (n=8)" in md
    assert "95% CI" in md
    for name in base.columns:
        if name in HIGHER_IS_BETTER:
            assert name in md


def test_failure_cases_returns_regressions_only():
    """Regressions are the reportable part; the helper must not surface wins as failures."""
    rows = make_rows(6)
    # tuned is worse on the last three: refusals instead of answers
    base = score_generations(rows, [f"Order {1000 + i} cancelled." for i in range(6)])
    tuned = score_generations(
        rows,
        [f"Order {1000 + i} cancelled." for i in range(3)]
        + ["I cannot help with that." for _ in range(3)],
    )
    worst = failure_cases(base, tuned, metric="hygiene_clean", k=10)
    assert len(worst) == 3
    assert (worst["delta"] < 0).all()


def test_save_comparison_writes_aggregate_and_per_example(tmp_path):
    """Per-example generations must be persisted, not just the aggregate.

    Both metric bugs in this project were found by reading raw outputs against an aggregate, and
    fixing them cost seconds instead of hours precisely because generations were on disk.
    """
    rows = make_rows(5)
    base = score_generations(rows, ["I cannot help." for _ in range(5)])
    tuned = score_generations(rows, [f"Order {1000 + i} cancelled." for i in range(5)])
    result = compare(base, tuned, slice_name="s", base_name="b", tuned_name="t")

    save_comparison(result, base, tuned, tmp_path / "out")

    for name in ("comparison.json", "comparison.md", "base_scores.parquet", "tuned_scores.parquet"):
        assert (tmp_path / "out" / name).exists(), f"{name} not written"
    reloaded = pd.read_parquet(tmp_path / "out" / "tuned_scores.parquet")
    assert "generation" in reloaded.columns
    assert len(reloaded) == 5
