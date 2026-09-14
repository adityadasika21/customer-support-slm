#!/usr/bin/env python
"""Full base-vs-fine-tuned comparison across every evaluation slice.

Produces the headline numbers. Base and tuned answer **byte-identical** rendered prompts, in the
same order, under identical greedy decoding; every metric is reported with a bootstrap confidence
interval and a win/tie/loss breakdown, and regressions are surfaced rather than averaged away.

    # in-process, base vs adapter
    python scripts/evaluate.py --model-key qwen3-1.7b --adapter artifacts/runs/final/adapter

    # against the running server, so the numbers describe what we actually ship
    python scripts/evaluate.py --model-key qwen3-1.7b --tuned-endpoint http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from csbot.data.loader import load_dataset_frame, load_ood_rows, select_rows
from csbot.eval.generate import DecodeParams, EndpointGenerator, HFGenerator
from csbot.eval.intent_clf import IntentClassifier, fit_intent_classifier
from csbot.eval.confusion import intent_category_map, intent_confusion
from csbot.eval.runner import (
    compare,
    failure_cases,
    render_prompts,
    save_comparison,
    score_generations,
)
from csbot.models.registry import FitTier, get as get_spec

LOG = logging.getLogger("evaluate")
VRAM_GB = 8.0

#: Slices evaluated by default.
#:
#: ``test_all`` (all 27 intents) rather than ``test_indomain`` (23), because the shipped model
#: trains under ``scope=final`` and has seen every intent. ``test_indomain`` exists to pair with
#: the research-scope pilots; using it for the shipped model silently omits four intents.
#:
#: ``test_heldout_intent`` is only a genuine unseen-intent test for a research-scope model. For
#: the shipped model those intents were trained on, so it is reported as a per-intent subset of
#: test_all rather than as a generalisation claim.
DEFAULT_SLICES = ("test_all", "ood")


def build_slices(df: pd.DataFrame, names: list[str], variant: str, limit: int | None) -> dict:
    slices: dict[str, pd.DataFrame] = {}
    for name in names:
        if name == "ood":
            rows = load_ood_rows(Path("eval_sets/ood_queries.jsonl"))
        else:
            rows = select_rows(df, slice_name=name, variant=variant)
            if limit and len(rows) > limit:
                # Stratify the cap by intent so a smaller run does not silently become a test of
                # whichever intents happen to sort first.
                per = max(1, limit // rows["intent"].nunique())
                # Concatenate per-group samples; groupby().apply() drops the grouping column in
                # pandas 3.0, which would silently remove "intent".
                parts = [
                    group.sample(min(len(group), per), random_state=17)
                    for _, group in rows.groupby("intent", sort=True)
                ]
                rows = pd.concat(parts).sort_values("id").reset_index(drop=True)
        slices[name] = rows
    return slices


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", required=True)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter for the tuned arm.")
    p.add_argument("--tuned-endpoint", default=None, help="Evaluate the served model over HTTP.")
    p.add_argument("--base-endpoint", default=None)
    p.add_argument("--endpoint-concurrency", type=int, default=24,
                   help="In-flight requests against the endpoint.")
    p.add_argument("--base-model-name", default=None,
                   help="Model name to request for the base arm (vLLM --served-model-name).")
    p.add_argument("--tuned-model-name", default=None,
                   help="Model name to request for the tuned arm, e.g. a --lora-modules name.")
    p.add_argument("--slices", nargs="+", default=list(DEFAULT_SLICES))
    p.add_argument("--variant", default="substituted", choices=["raw", "substituted"])
    p.add_argument("--limit", type=int, default=None, help="Cap rows per slice (stratified).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--load-in-4bit", type=lambda v: v.lower() in {"1", "true", "yes"},
                   default=None,
                   help="Override the precision. Default follows the model's SERVE tier, so the "
                        "numbers describe the weights that are actually shipped.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("reports/eval"))
    p.add_argument("--label", default="finetuned")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.adapter and not args.tuned_endpoint:
        raise SystemExit("provide --adapter or --tuned-endpoint")
    args.out.mkdir(parents=True, exist_ok=True)

    spec = get_spec(args.model_key)
    df = load_dataset_frame(Path("data/processed/dataset.parquet"))
    slices = build_slices(df, args.slices, args.variant, args.limit)
    LOG.info("slices: %s", {k: len(v) for k, v in slices.items()})

    clf_path = args.out / "intent_clf.joblib"
    if clf_path.exists():
        intent_clf = IntentClassifier.load(clf_path)
    else:
        intent_clf = fit_intent_classifier(select_rows(df, slice_name="train", variant=args.variant))
        intent_clf.save(clf_path)

    model_names = {"base": args.base_model_name, args.label: args.tuned_model_name}

    params = DecodeParams(max_new_tokens=args.max_new_tokens)
    # serve_tier, NOT fit_tier. fit_tier is the conservative combined verdict and is QUANTIZED
    # for granite because *training* needs 4-bit on 8 GB; inference does not, and the merged
    # weights we actually serve through vLLM are bf16. Evaluating in 4-bit would measure a model
    # nobody runs -- and it is slower here too (49 W and 2.2 GB of an 8 GB card, against 90 W
    # while training), because NF4 dequantisation dominates at these batch sizes.
    use_4bit = (args.load_in_4bit if args.load_in_4bit is not None
                else spec.serve_tier(VRAM_GB) is FitTier.QUANTIZED)
    LOG.info("evaluating in %s (serve tier for %s)", "4-bit" if use_4bit else "bf16", spec.key)

    # Prompts are rendered once per slice with the base tokenizer and reused for both arms, so
    # the two models provably receive identical bytes.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id)
    prompts = {name: render_prompts(rows, tokenizer, spec) for name, rows in slices.items()}

    def run_arm(label: str, adapter: Path | None, endpoint: str | None) -> dict:
        if endpoint:
            # The served model name is per-arm. vLLM can hold the base weights and a LoRA
            # adapter at once and route by name, so both arms answer from ONE process on an
            # 8 GB card -- two 4.8 GB servers would not fit, and loading them in sequence
            # doubles the wall clock for no benefit.
            served = model_names.get(label) or spec.repo_id
            # Concurrency measured, not guessed: on this card vLLM reaches 957 tok/s at 24
            # in-flight requests against 387 at 8, because KV cache -- not compute -- is the
            # binding constraint. Sending 8 leaves most of the paged cache idle.
            gen = EndpointGenerator(name=label, base_url=endpoint, model=served,
                                    concurrency=args.endpoint_concurrency, timeout=600.0)
            LOG.info("arm %s -> %s serving %s", label, endpoint, served)
            outputs = {n: gen.generate(prompts[n], params) for n in slices}
        else:
            gen = HFGenerator.load(
                spec.repo_id, name=label,
                adapter_dir=str(adapter) if adapter else None,
                load_in_4bit=use_4bit, batch_size=args.batch_size,
            )
            try:
                outputs = {n: gen.generate(prompts[n], params) for n in slices}
            finally:
                gen.unload()
                gc.collect()
                torch.cuda.empty_cache()
        return outputs

    LOG.info("=== base arm ===")
    base_out = run_arm("base", None, args.base_endpoint)
    LOG.info("=== tuned arm ===")
    tuned_out = run_arm(args.label, args.adapter, args.tuned_endpoint)

    summary = {}
    for name, rows in slices.items():
        with_ref = name != "ood"
        base_scores = score_generations(
            rows, base_out[name],
            intent_clf=intent_clf if with_ref else None, with_rouge=with_ref,
        )
        tuned_scores = score_generations(
            rows, tuned_out[name],
            intent_clf=intent_clf if with_ref else None, with_rouge=with_ref,
        )
        comparison = compare(
            base_scores, tuned_scores, slice_name=name,
            base_name=f"{spec.repo_id} (base)", tuned_name=args.label,
            intent_clf_ceiling=intent_clf.ceiling_accuracy if with_ref else None,
        )
        save_comparison(comparison, base_scores, tuned_scores, args.out / name)

        worst = failure_cases(base_scores, tuned_scores, metric="hygiene_clean", k=25)
        worst.to_csv(args.out / name / "regressions.csv", index=False)

        summary[name] = comparison.as_dict()
        print("\n" + comparison.to_markdown())
        print(f"  regressions where tuned < base on hygiene: {len(worst)}")

        # Intent confusion: a scalar accuracy cannot distinguish "fails at random" from
        # "cannot separate two near-synonymous intents", and those need different fixes.
        if with_ref:
            cats = intent_category_map(df)
            for label, frame in (("base", base_scores), ("tuned", tuned_scores)):
                rep = intent_confusion(frame, intent_clf, intent_to_category=cats)
                rep.matrix.to_csv(args.out / name / f"confusion_{label}.csv")
                summary[name][f"confusion_{label}"] = rep.as_dict()
                if label == "tuned":
                    print("\n#### intent confusion (fine-tuned)\n")
                    print(rep.to_markdown(max_rows=8))

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/summary.json")
    print(f"intent classifier ceiling on gold responses: {intent_clf.ceiling_accuracy:.3f}")


if __name__ == "__main__":
    main()
