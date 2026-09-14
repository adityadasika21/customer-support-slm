"""Evaluation orchestration: generate, score, and compare two models on identical prompts.

The comparison is strictly **paired** -- base and fine-tuned answer byte-identical rendered
prompts from the same slice, in the same order, under the same decoding parameters. Pairing lets
the bootstrap cancel per-example difficulty, which produces a much tighter and more honest
interval than comparing two independently-drawn means.

Nothing here chooses a single headline number. Each metric is reported with its own interval and
its own win/tie/loss breakdown, because they measure genuinely different things and a composite
would hide exactly the trade-off worth seeing -- most plausibly that the fine-tune wins decisively
on format while giving something back on open-ended queries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from csbot.data.loader import template_options
from csbot.eval.generate import DecodeParams, Generator
from csbot.eval.intent_clf import IntentClassifier
from csbot.eval.metrics import (
    bootstrap_ci,
    entity_fidelity,
    hygiene,
    paired_bootstrap,
    win_tie_loss,
)
from csbot.models.registry import ModelSpec
from csbot.serve.template import render_prompt

LOG = logging.getLogger(__name__)

#: Per-example metrics where a higher value is better. Used to orient win/tie/loss and to decide
#: the sign of a "regression" when reporting.
HIGHER_IS_BETTER = {
    "entity_recall": True,
    "entity_clean": True,
    "hygiene_clean": True,
    "intent_correct": True,
    "rouge_l": True,
    "think_leakage": False,
    "refusal": False,
    "boilerplate": False,
    "truncated": False,
    "repetition_rate": False,
    "n_hallucinated": False,
}


def render_prompts(rows: pd.DataFrame, tokenizer, spec: ModelSpec) -> list[str]:
    """Render the evaluation prompts once, so every model sees byte-identical input."""
    options = template_options(spec)
    return [
        render_prompt(tokenizer, row.instruction, options=options)
        for row in rows.itertuples(index=False)
    ]


def generate_for(
    generator: Generator,
    prompts: list[str],
    params: DecodeParams,
) -> list[str]:
    LOG.info("generating %d completions with %s", len(prompts), generator.name)
    return generator.generate(prompts, params)


@lru_cache(maxsize=1)
def _rouge_scorer():
    """Build the scorer once. Returns ``None`` if the optional dependency is absent."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:  # pragma: no cover - optional extra
        LOG.warning("rouge-score not installed; ROUGE-L will be reported as NaN")
        return None
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1, or NaN when the optional dependency is absent.

    Reported only as supporting evidence. On this dataset the references are long, florid
    paraphrases, so a good short answer scores badly and verbatim template regurgitation scores
    well -- the opposite of what we want to select for. See the module docstring in
    ``csbot.eval.metrics``.
    """
    scorer = _rouge_scorer()
    if scorer is None:
        return float("nan")
    return scorer.score(reference or "", candidate or "")["rougeL"].fmeasure


def score_generations(
    rows: pd.DataFrame,
    generations: list[str],
    *,
    intent_clf: IntentClassifier | None = None,
    with_rouge: bool = True,
) -> pd.DataFrame:
    """Compute every deterministic per-example metric. One row in, one row out."""
    if len(rows) != len(generations):
        raise ValueError(f"row/generation mismatch: {len(rows)} vs {len(generations)}")

    records = []
    for row, gen in zip(rows.itertuples(index=False), generations):
        ent = entity_fidelity(row.instruction, gen)
        hyg = hygiene(gen)
        records.append(
            {
                "id": row.id,
                "intent": row.intent,
                "category": row.category,
                "flags": row.flags,
                "instruction": row.instruction,
                "reference": row.response,
                "generation": gen,
                "entity_recall": ent.recall,
                "n_hallucinated": float(ent.n_hallucinated),
                "entity_clean": float(ent.clean),
                "hallucinated": ", ".join(ent.hallucinated),
                "think_leakage": float(hyg.think_leakage),
                "refusal": float(hyg.refusal),
                "boilerplate": float(hyg.boilerplate),
                "truncated": float(hyg.truncated),
                # Persisted in its own column, not just folded into hygiene_clean. This is the
                # defect that survived 14 metrics because it was never given a name of its own.
                "placeholder_leak": float(hyg.placeholder_leak),
                "empty_output": float(hyg.empty),
                "repetition_rate": hyg.repetition_rate,
                "n_words": float(hyg.n_words),
                "hygiene_clean": float(hyg.clean),
                "rouge_l": _rouge_l(row.response, gen) if with_rouge else float("nan"),
            }
        )

    out = pd.DataFrame.from_records(records)
    if intent_clf is not None:
        out["intent_correct"] = intent_clf.score_against(out["generation"], out["intent"])
    return out


@dataclass
class SliceComparison:
    """Base vs fine-tuned on one evaluation slice."""

    slice_name: str
    n: int
    base_name: str
    tuned_name: str
    metrics: dict
    """metric -> {base, tuned, diff, ci, p_value, significant, win/tie/loss}"""

    intent_clf_ceiling: float | None = None

    def as_dict(self) -> dict:
        return {
            "slice": self.slice_name,
            "n": self.n,
            "base": self.base_name,
            "tuned": self.tuned_name,
            "intent_clf_ceiling": self.intent_clf_ceiling,
            "metrics": self.metrics,
        }

    def to_markdown(self) -> str:
        lines = [
            f"### {self.slice_name} (n={self.n})",
            "",
            "| metric | base | fine-tuned | Δ [95% CI] | p | W/T/L |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, m in self.metrics.items():
            arrow = "↑" if HIGHER_IS_BETTER.get(name, True) else "↓"
            sig = "**" if m["significant"] else ""
            wtl = m["win_tie_loss"]
            lines.append(
                f"| {name} {arrow} | {m['base']:.3f} | {m['tuned']:.3f} | "
                f"{sig}{m['diff']:+.3f}{sig} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}] | "
                f"{m['p_value']:.3f} | {wtl['win']}/{wtl['tie']}/{wtl['loss']} |"
            )
        return "\n".join(lines)


def compare(
    base_scores: pd.DataFrame,
    tuned_scores: pd.DataFrame,
    *,
    slice_name: str,
    base_name: str,
    tuned_name: str,
    metrics: list[str] | None = None,
    intent_clf_ceiling: float | None = None,
    seed: int = 0,
) -> SliceComparison:
    """Paired comparison across every metric present in both frames."""
    if not base_scores["id"].equals(tuned_scores["id"]):
        raise ValueError("paired comparison requires identical ids in the same order")

    candidates = metrics or [m for m in HIGHER_IS_BETTER if m in base_scores.columns]
    results: dict[str, dict] = {}

    for name in candidates:
        if name not in base_scores.columns or name not in tuned_scores.columns:
            continue
        b = base_scores[name].to_numpy(dtype=float)
        t = tuned_scores[name].to_numpy(dtype=float)
        if np.isnan(b).all() or np.isnan(t).all():
            continue

        diff = paired_bootstrap(b, t, seed=seed)
        results[name] = {
            "base": diff.mean_base,
            "tuned": diff.mean_tuned,
            "base_ci": bootstrap_ci(b, seed=seed).as_dict(),
            "tuned_ci": bootstrap_ci(t, seed=seed).as_dict(),
            "diff": diff.diff,
            "ci": [diff.lo, diff.hi],
            "p_value": diff.p_value,
            "significant": diff.significant,
            "higher_is_better": HIGHER_IS_BETTER.get(name, True),
            "win_tie_loss": win_tie_loss(b, t),
        }

    return SliceComparison(
        slice_name=slice_name,
        n=len(base_scores),
        base_name=base_name,
        tuned_name=tuned_name,
        metrics=results,
        intent_clf_ceiling=intent_clf_ceiling,
    )


def save_comparison(
    comparison: SliceComparison,
    base_scores: pd.DataFrame,
    tuned_scores: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Persist the aggregate plus every per-example generation.

    The raw generations are kept deliberately: an aggregate nobody can drill into is not evidence,
    and the failure-case analysis is read straight out of these files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(comparison.as_dict(), indent=2))
    (out_dir / "comparison.md").write_text(comparison.to_markdown())
    base_scores.to_parquet(out_dir / "base_scores.parquet", index=False)
    tuned_scores.to_parquet(out_dir / "tuned_scores.parquet", index=False)
    LOG.info("wrote comparison to %s", out_dir)


def failure_cases(
    base_scores: pd.DataFrame,
    tuned_scores: pd.DataFrame,
    *,
    metric: str = "hygiene_clean",
    k: int = 20,
) -> pd.DataFrame:
    """Examples where the fine-tuned model did *worse* than the base model.

    Regressions are more informative than wins: they say what specialising cost us, and an honest report
    explicitly asks for failure cases.
    """
    merged = base_scores[["id", "intent", "instruction", metric, "generation"]].merge(
        tuned_scores[["id", metric, "generation"]],
        on="id", suffixes=("_base", "_tuned"),
    )
    merged["delta"] = merged[f"{metric}_tuned"] - merged[f"{metric}_base"]
    worse = merged[merged["delta"] < 0] if HIGHER_IS_BETTER.get(metric, True) else merged[merged["delta"] > 0]
    return worse.reindex(worse["delta"].abs().sort_values(ascending=False).index).head(k)
