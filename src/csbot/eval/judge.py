"""LLM-as-judge: blind, position-swapped, tie-allowing pairwise comparison.

Design follows Anthropic's evals guidance, with the places it is silent filled from standard
practice and marked as such:

* **Per-dimension rubrics, isolated judges** (their recommendation): "create clear, structured
  rubrics to grade each dimension of a task, and then grade each dimension with an isolated
  LLM-as-judge rather than using one to grade all dimensions." Each dimension is its own call.
* **Give the judge a way out** (theirs): every rubric permits TIE explicitly. Forcing a binary
  choice manufactures signal.
* **Calibrate against human judgment** (theirs, and the strongest one): judges are scored on
  agreement with human labels before any verdict of theirs is believed.
* **Blind, position-swapped, multi-judge** (standard practice, not from the article): the article
  lists pairwise comparison and multi-judge consensus as methods but gives no guidance on order
  effects, so the safeguards here are imported and labelled.

The judge is never assumed to work. ``validate`` runs it against pairs with a known answer and
against the human labels; a judge that fails is reported as failed rather than averaged in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VERDICTS = ("A", "B", "TIE")


@dataclass(frozen=True)
class Rubric:
    key: str
    name: str
    criterion: str


#: One rubric per dimension, worded so that two careful people would reach the same verdict --
#: the article's test for a good task.
RUBRICS: tuple[Rubric, ...] = (
    Rubric(
        "resolution", "Resolution",
        "Which response better addresses what the customer actually asked for? Prefer the one "
        "that identifies the request correctly and gives usable next steps. A reply that is "
        "polite but says nothing actionable does not win.",
    ),
    Rubric(
        "tone", "Support tone",
        "Which response reads more like a competent, empathetic human support agent? Consider "
        "warmth, clarity and professionalism. Length is not quality: padding, repetition and "
        "gushing are worse, not better.",
    ),
    Rubric(
        # Reworded after human calibration. The first version -- "which avoids inventing things"
        # -- was won by responses that avoided error by saying nothing: a content-free "please
        # consult our website" invents nothing and scored well. Two of three control misses came
        # from exactly that reading. Evasion is now explicitly not a way to win this dimension,
        # and the two failure modes measured in this project are named.
        "grounding", "Grounding",
        "Which response is more trustworthy about what it can and cannot know? Count against a "
        "response: inventing specifics it has no way to know (order status, balances, policies, "
        "contact details, delivery dates); showing the customer an unfilled template slot such "
        "as {{Order Number}}; and promising to perform an action it cannot actually perform, "
        "such as 'let me check that for you'. Saying that it cannot access something, and "
        "pointing to where the customer can, is good grounding. But a response that avoids "
        "these errors only by giving no real answer is NOT better grounded -- answer TIE in "
        "that case, since its failure belongs to Resolution.",
    ),
)

PROMPT = """You are evaluating two customer-support replies to the same message.

Customer message:
{message}

--- Response A ---
{a}

--- Response B ---
{b}

Question: {criterion}

Answer with exactly one word: A, B, or TIE. Answer TIE if they are equally good, or if you \
cannot tell. Do not explain.

Answer:"""


def build_prompt(message: str, a: str, b: str, rubric: Rubric) -> str:
    return PROMPT.format(message=message.strip(), a=a.strip(), b=b.strip(),
                         criterion=rubric.criterion)


_FIRST = re.compile(r"\b(A|B|TIE|TIED|NEITHER|BOTH|EQUAL|UNKNOWN)\b", re.I)


def parse_verdict(text: str) -> str | None:
    """First explicit verdict token, or ``None`` if the judge produced nothing usable.

    ``None`` is kept distinct from ``TIE``: an unparseable answer is a judge failure and belongs
    in the reliability figures, while TIE is a legitimate verdict. Collapsing them would let a
    judge that rambles look merely indecisive.
    """
    m = _FIRST.search(text or "")
    if not m:
        return None
    tok = m.group(1).upper()
    if tok in {"TIED", "NEITHER", "BOTH", "EQUAL", "UNKNOWN"}:
        return "TIE"
    return tok


def resolve(first: str | None, swapped: str | None) -> str:
    """Combine the two orderings into one verdict on the underlying pair.

    ``first`` judged (A=left, B=right); ``swapped`` judged the same pair with the sides
    exchanged, so its "A" means the right-hand response. A verdict counts only if it survives
    the swap; disagreement is recorded as a tie and counted separately as a position flip.

    This is the single most important safeguard here, and the article does not cover it: a judge
    with a strong position preference produces a confident, reproducible, meaningless result.
    """
    if first is None or swapped is None:
        return "UNPARSED"
    flip = {"A": "B", "B": "A", "TIE": "TIE"}
    if first == flip[swapped]:
        return first
    return "FLIP"
