#!/usr/bin/env python
"""Build the processed dataset: cluster, split, and materialise both ablation variants.

Writes a single ``dataset.parquet`` carrying every row with its cluster id, split assignment, and
both the raw and substituted text. Keeping one file (rather than a directory of per-split JSONL)
means the split assignment and the ablation variants can never drift out of sync, and the leakage
check can be re-run over the exact artefact that training consumed.

    python scripts/build_dataset.py --out data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from csbot.data.download import load_raw
from csbot.data.placeholders import (
    grounded_example,
    polished_example,
    raw_example,
    substitute_example,
)
from csbot.data.prepare import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SplitConfig,
    build_splits,
    check_leakage,
)

LOG = logging.getLogger("build_dataset")


def materialise_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns for both ablation arms.

    The substitution seed is the row id, so the generated entity values are deterministic and a
    rebuild reproduces byte-identical training data.
    """
    raw_i, raw_r, sub_i, sub_r, subs, retained = [], [], [], [], [], []
    grd_i, grd_r, grd_unmapped = [], [], []
    pol_i, pol_r, pol_unusable = [], [], []

    for row in df.itertuples(index=False):
        a = raw_example(row.instruction, row.response)
        b = substitute_example(row.instruction, row.response, seed=int(row.id))
        c = grounded_example(row.instruction, row.response, seed=int(row.id))
        e = polished_example(row.instruction, row.response, seed=int(row.id))
        raw_i.append(a.instruction)
        raw_r.append(a.response)
        sub_i.append(b.instruction)
        sub_r.append(b.response)
        subs.append(json.dumps(b.substituted))
        retained.append(json.dumps(b.retained))
        grd_i.append(c.instruction)
        grd_r.append(c.response)
        grd_unmapped.append(json.dumps(c.retained))
        pol_i.append(e.instruction)
        pol_r.append(e.response)
        pol_unusable.append(json.dumps(e.retained))

    out = df.copy()
    out["instruction_raw"] = raw_i
    out["response_raw"] = raw_r
    out["instruction_sub"] = sub_i
    out["response_sub"] = sub_r
    out["substituted"] = subs
    out["retained_placeholders"] = retained
    out["instruction_grd"] = grd_i
    out["response_grd"] = grd_r
    out["unmapped_placeholders"] = grd_unmapped

    # A grounded row is usable only if no slot survived. Marked rather than dropped here, so the
    # parquet stays one file describing every variant and the loader decides what each arm sees.
    out["instruction_pol"] = pol_i
    out["response_pol"] = pol_r
    out["grounded_usable"] = [u == "[]" for u in grd_unmapped]
    out["polished_usable"] = [u == "[]" for u in pol_unusable]
    return out


def summarise(df: pd.DataFrame) -> dict:
    """Split/intent/variant summary, written alongside the data for the report."""
    by_split = df.groupby("split").size().to_dict()
    heldout = df[df["heldout_intent"]]

    n_changed = int((df["instruction_raw"] != df["instruction_sub"]).sum())
    return {
        "rows": int(len(df)),
        "rows_by_split": {k: int(v) for k, v in by_split.items()},
        "split_fractions": {k: round(v / len(df), 4) for k, v in by_split.items()},
        "n_intents": int(df["intent"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "heldout_intents": sorted(heldout["intent"].unique().tolist()),
        "heldout_rows": int(len(heldout)),
        "research_train_rows": int(((df.split == "train") & ~df.heldout_intent).sum()),
        "final_train_rows": int((df.split == "train").sum()),
        "test_indomain_rows": int(((df.split == "test") & ~df.heldout_intent).sum()),
        "test_heldout_intent_rows": int(((df.split == "test") & df.heldout_intent).sum()),
        "substitution_changed_instructions": n_changed,
        "substitution_changed_pct": round(100 * n_changed / len(df), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--heldout-intents",
        nargs="*",
        default=None,
        help="Override the intents withheld from the research training set.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df.insert(0, "id", range(len(df)))
    LOG.info("loaded %d rows, %d intents", len(df), df["intent"].nunique())

    config_kwargs = {"seed": args.seed}
    if args.heldout_intents:
        config_kwargs["heldout_intents"] = tuple(args.heldout_intents)
    config = SplitConfig(**config_kwargs)

    split_df, stats = build_splits(df, config=config, threshold=args.threshold)

    leakage = check_leakage(split_df)
    if not leakage["leak_free"]:
        raise SystemExit(f"LEAKAGE DETECTED: {leakage}")
    LOG.info("leakage check passed: %d clusters, none spanning splits", leakage["n_clusters"])

    full = materialise_variants(split_df)
    out_path = args.out / "dataset.parquet"
    full.to_parquet(out_path, index=False)

    report = {
        "clustering": stats.as_dict(),
        "leakage": leakage,
        "splits": summarise(full),
        "config": {
            "threshold": args.threshold,
            "seed": args.seed,
            "heldout_intents": list(config.heldout_intents),
            "proportions": config.targets(),
        },
    }
    (args.out / "build_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path} ({len(full)} rows)")


if __name__ == "__main__":
    main()
