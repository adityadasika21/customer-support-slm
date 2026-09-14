#!/usr/bin/env python
"""Before fine-tuning: are the training targets actually better than the model you start from?

Supervised fine-tuning is imitation. It maximises the likelihood of the reference response, so it
moves the model **toward the targets** -- an improvement only if the targets are better than the
starting point. That condition is almost never checked, because the usual case is a weak base and
curated human data. Here it was inverted: an instruction-tuned, preference-aligned granite against
machine-generated templates.

Measured on this dataset, blind and position-swapped, the **gold references lost to the base
model** 27-3 on resolution, 25-4 on tone, 29-2 on grounding. Fine-tuning on them could not raise
overall answer quality; it could only trade quality for conformance. Every fit-to-reference metric
still improved -- ROUGE 0.214 -> 0.379, intent 0.756 -> 0.893 -- because those metrics ask
"does it resemble the target", which is the wrong question when the target is the problem.

This script is that check, as a gate. It costs one hour before training instead of a day after.

    # base model must be served (vLLM or any OpenAI-compatible endpoint)
    python scripts/check_targets_beat_base.py --n 40

Exit code 1 when the references lose, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csbot.data.loader import load_dataset_frame, select_rows  # noqa: E402
from csbot.eval.generate import DecodeParams, EndpointGenerator  # noqa: E402
from csbot.eval.runner import render_prompts  # noqa: E402
from csbot.models.registry import get as get_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-key", default="granite-3.3-2b")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001")
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--judge", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--variant", default="substituted")
    ap.add_argument("--seed", type=int, default=808)
    ap.add_argument("--out", type=Path, default=Path("reports/target_check"))
    args = ap.parse_args()

    spec = get_spec(args.model_key)
    rows = select_rows(load_dataset_frame("data/processed/dataset.parquet"),
                       slice_name="train", variant=args.variant)
    rows = rows[rows["response"].astype(str).str.len() > 120].sample(
        args.n, random_state=args.seed).reset_index(drop=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.repo_id)
    gen = EndpointGenerator(name="base", base_url=args.endpoint, model=args.base_name,
                            concurrency=16, timeout=600.0)
    base_answers = gen.generate(render_prompts(rows, tok, spec),
                                DecodeParams(max_new_tokens=512))

    # Blind, position randomised. "fine_tuned" is the canonical slot name the judge harness
    # counts under; here it holds the dataset's gold reference, and "base" the base model.
    rng = random.Random(args.seed)
    items, key = [], []
    for n, (row, base_text) in enumerate(zip(rows.itertuples(index=False), base_answers), 1):
        gold = str(row.response)
        flip = rng.random() < 0.5
        a, b = (gold, base_text) if flip else (base_text, gold)
        items.append({"item": n, "customer_message": str(row.instruction),
                      "response_a": a, "response_b": b})
        key.append({"item": n, "group": "random", "id": str(row.id), "intent": str(row.intent),
                    "a": "fine_tuned" if flip else "base",
                    "b": "base" if flip else "fine_tuned"})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs.json").write_text(json.dumps(items, indent=2))
    (args.out / "key.json").write_text(json.dumps(key, indent=2))

    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "judge_eval.py"),
         "--judge", args.judge, "--pairs", str(args.out / "pairs.json"),
         "--key", str(args.out / "key.json"), "--batch-size", "4",
         "--out", str(args.out / "judge.json")],
        check=True,
    )

    e = list(json.loads((args.out / "judge.json").read_text())["judges"].values())[0]
    print(f"\n{'dimension':13}{'reference':>11}{'base':>7}{'tie':>6}{'flip':>6}")
    ref_wins = base_wins = 0
    for d in ("resolution", "tone", "grounding"):
        p = e["preference"][d]
        ref_wins += p["fine_tuned"]
        base_wins += p["base"]
        print(f"{d:13}{p['fine_tuned']:>11}{p['base']:>7}{p['tie']:>6}{p['flip']:>6}")

    if e["flip_rate"] > 0.30:
        print(f"\nJUDGE UNRELIABLE HERE (flip rate {e['flip_rate']:.2f}) -- treat as no signal.")
        return 0

    print(f"\nreference wins {ref_wins}, base wins {base_wins}")
    if ref_wins >= base_wins:
        print("The targets are at least as good as the base model. SFT can help.")
        return 0
    print(
        "\nTHE TARGETS ARE WORSE THAN THE MODEL YOU ARE STARTING FROM.\n"
        "Plain SFT will move the model toward them and reduce overall answer quality, while\n"
        "every fit-to-reference metric (ROUGE, and any classifier trained on this data) rises.\n"
        "Options: rewrite the targets with a stronger model; preference-tune instead of\n"
        "imitating; train briefly for format only; or do not fine-tune."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
