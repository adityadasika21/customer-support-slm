"""Freeze the in-domain anchor set the off-topic rail measures distance from.

The rail asks "is this query unlike every real support query we have seen?". That needs a
reference set of real support queries. Sampling it at serve time would couple serving to the
training parquet and make the rail's behaviour depend on a random seed at boot, so the anchors are
frozen here into ``guardrails/anchors.json`` and shipped with the rails.

Sampling is stratified across all 27 intents. An unstratified sample would over-represent the
common intents, leaving the rare ones far from any anchor -- which would show up in production as
the rail refusing exactly the customers with the unusual problems.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=Path("data/processed/dataset.parquet"))
    ap.add_argument("--per-intent", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("guardrails/anchors.json"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--probes", type=Path, default=Path("eval_sets/ood_queries.jsonl"))
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    train = df[df["split"] == "train"]
    col = "instruction_sub" if "instruction_sub" in train.columns else "instruction"

    parts = [
        g.sample(min(len(g), args.per_intent), random_state=args.seed)
        for _, g in train.groupby("intent")
    ]
    sample = pd.concat(parts)

    # Drop anchors that resemble a held-out probe. The anchors are real training queries, so a
    # probe looking like one is usually harmless -- but the off-topic rail decides by distance
    # to these, and an anchor sitting on top of a probe scores that probe in-domain by
    # construction. Cheap to exclude, and it keeps check_rails_leakage.py a hard gate rather
    # than one with a documented exception.
    import json as _json

    from csbot.data.guardrails import _norm

    probes = [
        _json.loads(line)["query"]
        for line in args.probes.read_text().splitlines() if line.strip()
    ] if args.probes.exists() else []
    probe_tokens = [set(_norm(q).split()) for q in probes]

    def collides(text: str) -> bool:
        t = set(_norm(text).split())
        if not t:
            return False
        return any(pt and len(t & pt) / len(t | pt) >= 0.6 for pt in probe_tokens)

    before = len(sample)
    sample = sample[~sample[col].map(collides)]
    if before != len(sample):
        print(f"dropped {before - len(sample)} anchors that resemble a held-out probe")

    payload = {
        "description": (
            "Real in-domain customer queries, stratified across every intent. The off-topic rail "
            "flags a request when it is further from all of these than a tuned threshold."
        ),
        "source": str(args.parquet),
        "column": col,
        "per_intent": args.per_intent,
        "seed": args.seed,
        "n_intents": int(sample["intent"].nunique()),
        "anchors": sample[col].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"{len(payload['anchors'])} anchors across {payload['n_intents']} intents -> {args.out}")


if __name__ == "__main__":
    main()
