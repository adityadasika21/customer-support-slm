"""Intent fidelity: does the generated reply address the right kind of request?

A cheap, deterministic proxy for "did it answer the question". We fit a TF-IDF + logistic
regression classifier on gold ``response -> intent`` pairs, then apply it to *generated* replies
and compare the predicted intent against the example's gold intent.

Why this is a fair measure, and where it is weak
------------------------------------------------
The classifier only ever sees the **train** split, so it never learned from the responses it is
scoring. It is a fixed, model-agnostic instrument applied identically to base and fine-tuned
outputs, which is what makes the comparison meaningful even though the absolute number is only a
proxy.

Its ceiling is the separability of the intents themselves -- ``get_invoice`` and ``check_invoice``
are genuinely close -- so we report the classifier's own held-out accuracy on gold responses
alongside the model scores. That number is the practical upper bound: a model scoring near it is
doing as well as this instrument can detect, and further movement means nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class IntentClassifier:
    """Wraps the fitted pipeline plus the diagnostics needed to interpret its scores."""

    pipeline: object
    labels: tuple[str, ...]
    ceiling_accuracy: float
    """Accuracy on held-out *gold* responses -- the instrument's own reliability, and the
    practical upper bound for any generation score."""

    n_train: int

    def predict(self, texts) -> np.ndarray:
        return self.pipeline.predict(list(texts))

    def score_against(self, texts, gold_intents) -> np.ndarray:
        """Per-example 0/1 correctness, so it can be bootstrapped like any other metric."""
        preds = self.predict(texts)
        return (preds == np.asarray(list(gold_intents))).astype(float)

    def save(self, path: Path | str) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "labels": self.labels,
                "ceiling_accuracy": self.ceiling_accuracy,
                "n_train": self.n_train,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> IntentClassifier:
        import joblib

        blob = joblib.load(Path(path))
        return cls(
            pipeline=blob["pipeline"],
            labels=tuple(blob["labels"]),
            ceiling_accuracy=blob["ceiling_accuracy"],
            n_train=blob["n_train"],
        )


def fit_intent_classifier(
    train_rows: pd.DataFrame,
    *,
    seed: int = 17,
    holdout_frac: float = 0.15,
) -> IntentClassifier:
    """Fit the classifier on gold responses from the training split only.

    ``train_rows`` must come from ``split == "train"``. Fitting on validation or test responses
    would let the instrument learn the very text it is later asked to judge.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    if not (train_rows["split"] == "train").all():
        raise ValueError("intent classifier must be fitted on the train split only")

    X = train_rows["response"].tolist()
    y = train_rows["intent"].to_numpy()

    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=holdout_frac, random_state=seed, stratify=y
    )

    pipeline = make_pipeline(
        # Word unigrams+bigrams: responses are long and fluent, so word features are informative
        # and far cheaper than the character n-grams the short, typo-laden instructions needed.
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed),
    )
    pipeline.fit(X_fit, y_fit)

    ceiling = float((pipeline.predict(X_hold) == y_hold).mean())
    LOG.info(
        "intent classifier: %d train rows, %d classes, gold-response accuracy %.3f",
        len(X_fit), len(set(y)), ceiling,
    )

    return IntentClassifier(
        pipeline=pipeline,
        labels=tuple(sorted(set(y))),
        ceiling_accuracy=ceiling,
        n_train=len(X_fit),
    )
