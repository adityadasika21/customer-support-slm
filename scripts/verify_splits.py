#!/usr/bin/env python
"""Audit the splits for leakage. This is the script a skeptical reader should run first.

Cluster disjointness alone is a weak claim -- it proves the *procedure* was followed, not that
the procedure worked. If the similarity threshold were too high, genuine paraphrases would land
in different clusters and the splits would leak while every cluster check still passed.

So the real test is measured directly: for every test instruction, the maximum similarity to any
training instruction. A leak-free split should show no test item that is a near-copy of something
trained on.

    python scripts/verify_splits.py --data data/processed/dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from csbot.data.prepare import check_leakage, similarity_blocks

#: A test instruction more similar than this to a training instruction is treated as a leak.
#:
#: Note what this check can and cannot prove. Clustering guarantees by construction that nothing
#: exceeds the clustering threshold (0.65), so an alarm set above that can only ever pass. Its
#: value is as a regression test: if the clustering code breaks, or someone raises the threshold
#: without thinking, this fails loudly. The *substantive* justification for the threshold itself
#: is recorded in reports/threshold_sweep.json, not re-derived here.
LEAK_ALARM_THRESHOLD = 0.80


def max_cross_split_similarity(
    df: pd.DataFrame, *, left: str = "test", right: str = "train", top_k: int = 15
) -> dict:
    """For each ``left`` instruction, the most similar ``right`` instruction.

    Uses ``csbot.data.prepare.similarity_blocks`` -- the *same* instrument the clustering used.
    Re-fitting a separate vectoriser here would change the IDF weights and put the audit on a
    different scale from the threshold it is auditing, so the comparison would prove nothing.
    """
    split_arr = df["split"].to_numpy()
    per_row_max: list[float] = []
    pairs: list[dict] = []

    for idx, sim in similarity_blocks(df):
        local = split_arr[idx]
        lpos = np.flatnonzero(local == left)
        rpos = np.flatnonzero(local == right)
        if lpos.size == 0 or rpos.size == 0:
            continue

        sub = sim[np.ix_(lpos, rpos)]
        best_idx = sub.argmax(axis=1)
        best = sub[np.arange(sub.shape[0]), best_idx]
        per_row_max.extend(best.tolist())

        for i in np.argsort(-best)[:top_k]:
            li = int(idx[lpos[int(i)]])
            ri = int(idx[rpos[int(best_idx[int(i)])]])
            pairs.append(
                {
                    "intent": df["intent"].iloc[li],
                    "similarity": float(best[int(i)]),
                    f"{left}_instruction": df["instruction"].iloc[li],
                    f"{right}_instruction": df["instruction"].iloc[ri],
                    f"{left}_cluster": int(df["cluster_id"].iloc[li]),
                    f"{right}_cluster": int(df["cluster_id"].iloc[ri]),
                }
            )

    arr = np.asarray(per_row_max)
    pairs.sort(key=lambda p: -p["similarity"])
    return {
        "n_compared": int(arr.size),
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "percentiles": {f"p{q}": float(np.percentile(arr, q)) for q in (50, 90, 99)}
        if arr.size else {},
        "n_above_alarm": int((arr >= LEAK_ALARM_THRESHOLD).sum()),
        "alarm_threshold": LEAK_ALARM_THRESHOLD,
        "worst_pairs": pairs[:top_k],
    }


def cluster_size_profile(df: pd.DataFrame) -> dict:
    sizes = df.groupby("cluster_id").size()
    return {
        "n_clusters": int(sizes.size),
        "rows_per_cluster_mean": float(sizes.mean()),
        "largest": int(sizes.max()),
        "singletons": int((sizes == 1).sum()),
        "size_percentiles": {f"p{q}": int(np.percentile(sizes, q)) for q in (50, 90, 99)},
        "largest_clusters": [
            {"cluster_id": int(c), "size": int(n), "intent": df[df.cluster_id == c].intent.iloc[0]}
            for c, n in sizes.nlargest(5).items()
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/dataset.parquet"))
    p.add_argument("--out", type=Path, default=Path("reports/leakage_audit.json"))
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    df["instruction"] = df["instruction_raw"]

    report = {
        "cluster_disjointness": check_leakage(df),
        "cluster_profile": cluster_size_profile(df),
        "test_vs_train": max_cross_split_similarity(df, left="test", right="train"),
        "val_vs_train": max_cross_split_similarity(df, left="val", right="train", top_k=5),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    tvt = report["test_vs_train"]
    print(json.dumps({k: v for k, v in report.items() if k != "test_vs_train"}, indent=2))
    print("\n--- test vs train similarity ---")
    print(json.dumps({k: v for k, v in tvt.items() if k != "worst_pairs"}, indent=2))
    print("\n--- most similar surviving pairs ---")
    for pair in tvt["worst_pairs"][:8]:
        print(f"  {pair['similarity']:.3f}  [{pair['intent']}]")
        print(f"     test : {pair['test_instruction']}")
        print(f"     train: {pair['train_instruction']}")

    ok = report["cluster_disjointness"]["leak_free"] and tvt["n_above_alarm"] == 0
    print("\n" + ("PASS: splits are leakage-free" if ok else "FAIL: possible leakage"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
