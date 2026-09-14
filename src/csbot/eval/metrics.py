"""Deterministic metrics for base-vs-tuned comparison.

Design stance
-------------
The obvious metric here -- ROUGE/BERTScore against the reference response -- is close to a trap on
this dataset, and we deliberately do not lead with it. The references are long, florid, and
paraphrase-generated: a genuinely good short answer that differs in wording scores *badly*, while
a model that regurgitates the template verbatim scores *brilliantly*. Since template regurgitation
is precisely the failure mode we are trying to detect, optimising for reference overlap would
select for the thing we are trying to avoid. Those metrics are reported as supporting evidence,
explicitly caveated.

The metrics that actually bind are behavioural:

* **entity fidelity** -- does the reply echo the order number it was given, and does it refrain
  from inventing one? This is the dangerous real-world failure: a support bot that confidently
  quotes a fabricated order number.
* **format hygiene** -- reasoning-trace leakage, degenerate repetition, refusals, assistant
  boilerplate, truncation. This is where base models typically lose badly and a fine-tune wins
  clearly and legibly.
* **intent fidelity** -- did the reply address the right *kind* of request (see
  ``csbot.eval.intent_clf``).

Every aggregate is reported with a bootstrap confidence interval. On a few hundred examples a
raw delta of "68% vs 61%" is not distinguishable from noise, and a skeptical engineer will say so.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

# --------------------------------------------------------------------------------------------
# Entity handling
# --------------------------------------------------------------------------------------------

PLACEHOLDER_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")

#: Identifier-shaped tokens: things a customer would recognise as "their" number. Requires at
#: least two digits so ordinary prose numbers ("2 days", "24 hours") do not register as entities.
#:
#: The boundary conditions are fiddly and were wrong at first. An earlier lookahead of ``(?![\w.])``
#: excluded any identifier followed by a period -- which is to say, every identifier at the end of
#: a sentence, precisely where a hallucinated order number usually appears. The rule that is
#: actually wanted is "not part of a decimal": reject ``49`` in ``$49.99`` but accept ``55512`` in
#: ``...your order 55512.``
IDENTIFIER_RE = re.compile(
    r"(?<!\w)(?<!\d\.)"            # not mid-word, not the tail of a decimal
    r"(?:#\s?)?[A-Z]{0,4}[-/]?"    # optional hash and letter prefix (#A-7781, ORD-2291)
    r"\d{2,}(?:[-/]\d{2,})*"       # at least two digits, optionally hyphen/slash grouped
    r"(?!\w)(?!\.\d)"              # not mid-word, not followed by a decimal fraction
)

#: Durations, quantities and similar prose numerals, which are legitimately generated rather than
#: copied and must not be counted as hallucinated identifiers.
_PROSE_NUMBER_RE = re.compile(
    r"\b\d{1,3}\s*(?:-|–|to\s+)?\s*\d{0,3}\s*"
    r"(?:business\s+)?(?:second|minute|hour|day|week|month|year|%|percent|item|time)s?\b",
    re.I,
)


def _strip_prose_numbers(text: str) -> str:
    return _PROSE_NUMBER_RE.sub(" ", text or "")


def normalize_entity(token: str) -> str:
    """Canonical form for comparing identifiers across surface variations (``#12345`` vs ``12345``)."""
    return re.sub(r"[^a-z0-9]", "", (token or "").lower())


def extract_entities(text: str) -> set[str]:
    """Identifier-like entities in ``text``: placeholders and identifier-shaped tokens."""
    text = text or ""
    found = {f"{{{{{name}}}}}" for name in PLACEHOLDER_RE.findall(text)}
    cleaned = _strip_prose_numbers(PLACEHOLDER_RE.sub(" ", text))
    found |= {m.group(0).strip() for m in IDENTIFIER_RE.finditer(cleaned)}
    return {e for e in found if normalize_entity(e)}


@dataclass
class EntityScore:
    """Per-example entity handling."""

    n_expected: int
    n_echoed: int
    n_hallucinated: int
    recall: float
    """Share of the customer's identifiers that appear in the reply."""

    hallucinated: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """No invented identifiers. The property that actually matters in production."""
        return self.n_hallucinated == 0


def is_placeholder(token: str) -> bool:
    """Whether an entity is a ``{{Template Slot}}`` rather than a concrete value."""
    return bool(PLACEHOLDER_RE.fullmatch((token or "").strip()))


def entity_fidelity(instruction: str, generated: str) -> EntityScore:
    """Compare identifiers in the reply against those the customer supplied.

    *Recall* rewards echoing the customer's own reference back to them.

    *Hallucination* counts only **concrete** identifiers the reply invented -- an order number
    the customer never gave. Placeholders are deliberately excluded: ``{{Customer Support Phone
    Number}}`` is a "fill this in" marker, not a fabricated fact, and emitting one for
    information the model cannot know is exactly the behaviour the data pipeline is designed to
    teach (see ``csbot.data.placeholders``).

    Counting placeholders as hallucinations inverted this metric. Measured on the funnel subset:
    the **gold reference responses** scored 0.674 clean, while an untuned base model scored 0.957
    purely because it never emits placeholders at all -- so the metric rewarded the model that
    had learned *less*. The fine-tuned model's 0.717 is in fact above the reference ceiling.
    """
    expected = {normalize_entity(e): e for e in extract_entities(instruction)}
    produced = {normalize_entity(e): e for e in extract_entities(generated)}

    echoed = set(expected) & set(produced)
    invented = {
        k for k in set(produced) - set(expected) if not is_placeholder(produced[k])
    }

    return EntityScore(
        n_expected=len(expected),
        n_echoed=len(echoed),
        n_hallucinated=len(invented),
        recall=len(echoed) / len(expected) if expected else float("nan"),
        hallucinated=tuple(sorted(produced[k] for k in invented)),
    )


# --------------------------------------------------------------------------------------------
# Format hygiene
# --------------------------------------------------------------------------------------------

_THINK_RE = re.compile(r"<\s*/?\s*think\s*>|<\s*/?\s*reasoning\s*>", re.I)

#: Refusal means "I will not / cannot help with this", NOT "I am sorry this happened to you".
#:
#: The first version of this pattern matched a bare ``I'm sorry`` and reported a 59% refusal rate
#: for a model whose replies were in fact well-formed, empathetic support answers. In this domain
#: apology is the desired register -- "I'm sorry for the inconvenience" is what a good agent
#: writes -- so matching it penalised exactly the tone we are training for.
#:
#: The fix is to require an explicit statement of *inability*, optionally preceded by an apology,
#: rather than treating the apology itself as evidence.
_REFUSAL_RE = re.compile(
    r"(?:"
    r"\bi\s+(?:can(?:no|')?t|cannot|am\s+unable\s+to|'m\s+unable\s+to)\s+"
    r"(?:help|assist|provide|do|answer|access|give|share|disclose)"
    r"|\bi\s+(?:do\s*n(?:o|')t|don't)\s+have\s+(?:access|the\s+ability|permission|that\s+information)"
    r"|\bas\s+an\s+ai(?:\s+language\s+model)?\b"
    r"|\bi(?:'m|\s+am)\s+(?:sorry|afraid)[,]?\s+(?:but\s+)?i\s+(?:can(?:no|')?t|cannot|am\s+unable)"
    r"|\bunfortunately[,]?\s+i\s+(?:can(?:no|')?t|cannot|am\s+unable)"
    r"|\bi'?m\s+not\s+able\s+to\s+(?:help|assist|provide|do|answer|access)"
    r")",
    re.I,
)
_BOILERPLATE_RE = re.compile(
    r"\b(?:as an ai language model|i'?m just an ai|i am an ai assistant"
    r"|here(?:'s| is) (?:a|the) (?:possible |sample |example )?(?:response|reply|answer))\b",
    re.I,
)
_SENTENCE_END_RE = re.compile(r"[.!?\"')\]]\s*$")


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeats. High values indicate degenerate looping."""
    tokens = (text or "").split()
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return 1.0 - (len(counts) / len(grams))


def looks_truncated(text: str) -> bool:
    """Heuristic: the reply stops mid-sentence, usually meaning it hit the token cap."""
    text = (text or "").strip()
    return bool(text) and not _SENTENCE_END_RE.search(text)


@dataclass
class HygieneScore:
    """Format-level defects. Every field is a defect, so lower is better throughout."""

    think_leakage: bool
    """Reasoning trace escaped into the customer-facing reply. A production incident."""

    refusal: bool
    boilerplate: bool
    truncated: bool
    empty: bool
    repetition_rate: float
    n_chars: int
    n_words: int

    placeholder_leak: bool = False
    """An unfilled template slot was shown to the customer, e.g. ``{{Customer Support Phone
    Number}}``.

    Added after a human labelling 20 blind pairs found what 14 automatic metrics missed: the
    fine-tuned model emitted a raw slot in **30.5%** of held-out responses against 0.1% for the
    base model. It was invisible because ``is_placeholder`` had been excluded from the
    hallucination count -- a correct fix for a different bug that left this one unmeasured.

    It counts against ``clean``: text containing literal braces is not a serviceable reply,
    whatever else is right about it. Including it lowers previously reported hygiene figures,
    which is the point -- those figures were measuring around the defect."""

    @property
    def clean(self) -> bool:
        return not (
            self.think_leakage
            or self.refusal
            or self.boilerplate
            or self.truncated
            or self.empty
            or self.placeholder_leak
            or self.repetition_rate > 0.30
        )


#: A literal template slot reaching the customer. Deliberately matches the brace syntax itself
#: rather than a list of known names, so a slot type never seen in training still counts.
_PLACEHOLDER_LEAK_RE = re.compile(r"\{\{\s*[^{}]+\s*\}\}")


def hygiene(text: str) -> HygieneScore:
    text = text or ""
    stripped = text.strip()
    return HygieneScore(
        think_leakage=bool(_THINK_RE.search(text)),
        refusal=bool(_REFUSAL_RE.search(text)),
        boilerplate=bool(_BOILERPLATE_RE.search(text)),
        truncated=looks_truncated(text),
        empty=not stripped,
        repetition_rate=repetition_rate(text),
        n_chars=len(stripped),
        n_words=len(stripped.split()),
        placeholder_leak=bool(_PLACEHOLDER_LEAK_RE.search(text)),
    )


# --------------------------------------------------------------------------------------------
# Aggregation with uncertainty
# --------------------------------------------------------------------------------------------


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    def as_dict(self) -> dict:
        return asdict(self)


def bootstrap_ci(
    values, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap CI for the mean, ignoring NaNs.

    Non-parametric because these metrics are bounded, skewed and often binary -- a normal
    approximation would be wrong at exactly the interesting extremes.
    """
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if arr.size == 1:
        return Interval(float(arr[0]), float(arr[0]), float(arr[0]), 1)

    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(arr.mean()), float(lo), float(hi), int(arr.size))


@dataclass
class PairedDiff:
    """A paired base-vs-tuned comparison on identical prompts."""

    mean_base: float
    mean_tuned: float
    diff: float
    lo: float
    hi: float
    n: int
    p_value: float
    """Two-sided bootstrap p-value: the share of resamples whose sign contradicts the observed
    difference, doubled. Not a t-test -- these distributions are not normal."""

    @property
    def significant(self) -> bool:
        """The CI for the difference excludes zero."""
        return (self.lo > 0) or (self.hi < 0)

    def as_dict(self) -> dict:
        return asdict(self) | {"significant": self.significant}


def paired_bootstrap(
    base, tuned, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> PairedDiff:
    """Bootstrap the *difference* on paired observations.

    Pairing matters: base and tuned answer byte-identical prompts, so resampling examples (rather
    than the two arms independently) cancels per-example difficulty and gives a much tighter,
    more honest interval.
    """
    b = np.asarray(base, dtype=float)
    t = np.asarray(tuned, dtype=float)
    if b.shape != t.shape:
        raise ValueError(f"paired arrays must align: {b.shape} vs {t.shape}")

    keep = ~(np.isnan(b) | np.isnan(t))
    b, t = b[keep], t[keep]
    if b.size == 0:
        return PairedDiff(*( [float("nan")] * 5 ), 0, float("nan"))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b.size, size=(n_boot, b.size))
    diffs = (t[idx] - b[idx]).mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])

    observed = float(t.mean() - b.mean())
    tail = float((diffs <= 0).mean() if observed > 0 else (diffs >= 0).mean())
    return PairedDiff(
        mean_base=float(b.mean()),
        mean_tuned=float(t.mean()),
        diff=observed,
        lo=float(lo),
        hi=float(hi),
        n=int(b.size),
        p_value=min(1.0, 2 * tail),
    )


def win_tie_loss(base, tuned, *, tolerance: float = 0.0) -> dict[str, int]:
    """Per-example win/tie/loss counts.

    Reported alongside means because a mean can hide the shape of an improvement: winning hugely
    on a third of cases while losing mildly on the rest is a very different model from winning
    slightly everywhere, and the two demand different decisions.
    """
    b = np.asarray(base, dtype=float)
    t = np.asarray(tuned, dtype=float)
    keep = ~(np.isnan(b) | np.isnan(t))
    b, t = b[keep], t[keep]
    delta = t - b
    return {
        "win": int((delta > tolerance).sum()),
        "tie": int((np.abs(delta) <= tolerance).sum()),
        "loss": int((delta < -tolerance).sum()),
        "n": int(b.size),
    }
