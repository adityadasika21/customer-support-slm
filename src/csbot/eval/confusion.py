"""Intent confusion analysis: which intents get mistaken for which, and does it matter.

A scalar intent accuracy hides the thing you actually want to know. "0.900" is compatible with
two very different models: one that fails uniformly at random across 27 intents, and one that is
perfect except that it cannot separate ``get_invoice`` from ``check_invoice``. The second is a
far better support bot, and the fix for it is a data problem rather than a training problem.

So errors are reported as a matrix, and additionally split into two kinds:

**Near-miss** -- confused with an intent in the *same category*. ``get_invoice`` answered as
``check_invoice`` is a customer who still gets a useful reply about their invoice. These are
partly an artefact of the taxonomy: the dataset's own categories group intents that overlap
semantically, and no model (nor the classifier measuring it) can be expected to split them
cleanly.

**Cross-category** -- confused with an intent in a *different* category. A refund question
answered as a shipping question is a genuine failure, and is what the headline number should be
sensitive to.

Reported alongside the intent classifier's own ceiling on gold responses, because a model cannot
be distinguished from the instrument beyond that point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfusionReport:
    """Intent confusions for one set of generations."""

    matrix: pd.DataFrame
    """Rows = true intent, columns = predicted intent, values = counts."""

    accuracy: float
    near_miss_rate: float
    """Share of *all* examples confused within the same category."""

    cross_category_rate: float
    """Share of *all* examples confused across categories -- the errors that matter."""

    top_confusions: pd.DataFrame
    per_intent: pd.DataFrame

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "near_miss_rate": round(self.near_miss_rate, 4),
            "cross_category_rate": round(self.cross_category_rate, 4),
            "top_confusions": self.top_confusions.to_dict(orient="records"),
            "worst_intents": self.per_intent.head(8).to_dict(orient="records"),
        }

    def to_markdown(self, max_rows: int = 10) -> str:
        lines = [
            f"**accuracy {self.accuracy:.3f}**. The remaining "
            f"{self.near_miss_rate + self.cross_category_rate:.3f} splits into "
            f"**{self.near_miss_rate:.3f} near-miss** (confused with a sibling intent in the "
            f"same category) and **{self.cross_category_rate:.3f} cross-category** (a genuinely "
            f"wrong kind of answer).",
            "",
            "Most frequent confusions:",
            "",
            "| true intent | predicted as | n | same category? |",
            "| --- | --- | --- | --- |",
        ]
        for r in self.top_confusions.head(max_rows).itertuples(index=False):
            same = "near-miss" if r.same_category else "**cross-category**"
            lines.append(f"| `{r.true_intent}` | `{r.predicted_intent}` | {r.n} | {same} |")
        lines += [
            "",
            "Weakest intents:",
            "",
            "| intent | n | recall | most often mistaken for |",
            "| --- | --- | --- | --- |",
        ]
        for r in self.per_intent.head(max_rows).itertuples(index=False):
            lines.append(
                f"| `{r.intent}` | {r.n} | {r.recall:.3f} | "
                f"`{r.top_confusion}`{'' if r.top_confusion == '-' else f' ({r.top_confusion_n})'} |"
            )
        return "\n".join(lines)


def intent_confusion(
    scored: pd.DataFrame,
    intent_clf,
    *,
    intent_to_category: dict[str, str],
) -> ConfusionReport:
    """Build the confusion report from a scored generation frame.

    ``scored`` needs ``intent`` (gold) and ``generation``; predictions come from the same fixed
    classifier used for the headline metric, so the two numbers cannot disagree.
    """
    true = scored["intent"].to_numpy()
    pred = intent_clf.predict(scored["generation"].fillna(""))

    labels = sorted(set(true) | set(pred))
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(true, pred):
        matrix.loc[t, p] += 1

    correct = true == pred
    same_cat = np.array(
        [
            intent_to_category.get(t) == intent_to_category.get(p)
            for t, p in zip(true, pred)
        ]
    )
    n = len(true)

    rows = []
    for t, p in zip(true, pred):
        if t != p:
            rows.append((t, p))
    conf = (
        pd.DataFrame(rows, columns=["true_intent", "predicted_intent"])
        .value_counts()
        .reset_index(name="n")
        if rows
        else pd.DataFrame(columns=["true_intent", "predicted_intent", "n"])
    )
    if not conf.empty:
        conf["same_category"] = [
            intent_to_category.get(t) == intent_to_category.get(p)
            for t, p in zip(conf["true_intent"], conf["predicted_intent"])
        ]

    per_intent = []
    for intent in sorted(set(true)):
        mask = true == intent
        sub_pred = pred[mask]
        wrong = sub_pred[sub_pred != intent]
        top, top_n = ("-", 0)
        if len(wrong):
            vals, counts = np.unique(wrong, return_counts=True)
            top, top_n = str(vals[counts.argmax()]), int(counts.max())
        per_intent.append(
            {
                "intent": intent,
                "n": int(mask.sum()),
                "recall": float((sub_pred == intent).mean()),
                "top_confusion": top,
                "top_confusion_n": top_n,
            }
        )
    per_intent_df = pd.DataFrame(per_intent).sort_values("recall")

    return ConfusionReport(
        matrix=matrix,
        accuracy=float(correct.mean()),
        near_miss_rate=float((~correct & same_cat).sum() / n),
        cross_category_rate=float((~correct & ~same_cat).sum() / n),
        top_confusions=conf,
        per_intent=per_intent_df,
    )


def intent_category_map(df: pd.DataFrame) -> dict[str, str]:
    """Intent -> category, taken from the dataset itself rather than hardcoded."""
    return dict(df.groupby("intent")["category"].first())
