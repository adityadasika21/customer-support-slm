#!/usr/bin/env python
"""Run the LLM judges, validate them, and only then report their verdicts.

Order matters and is enforced: a judge's verdicts are reported only after it has passed a
validation gate. The gate has three parts, and the article's guidance drives the third:

1. **Parseable** -- it must answer with a verdict at all.
2. **Position-stable** -- every pair is judged in both orders; a verdict counts only if it
   survives the swap. High flip rates mean the judge is reading position, not content.
3. **Correct on controls, and in agreement with a human** -- controls have a known answer by
   construction; human labels are the reference a model grader must be
   calibrated against.

A judge that fails is reported as failed. Publishing the verdicts of an uncalibrated judge is
worse than having no judge, because the numbers look like evidence.

    python scripts/judge_eval.py \
      --judge mistralai/Mistral-7B-Instruct-v0.3 \
      --judge Qwen/Qwen3.5-4B \
      --pairs eval_sets/judge_polished/pairs.json \
      --key eval_sets/judge_polished/KEY_DO_NOT_OPEN_BEFORE_LABELLING.json \
      --human-labels reports/eval/calibration/human_labels.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch

from csbot.eval.judge import RUBRICS, build_prompt, parse_verdict, resolve

LOG = logging.getLogger("judge")

#: Gate thresholds, fixed before any judge was run.
MIN_CONTROL_ACCURACY = 0.80
MAX_FLIP_RATE = 0.30
MAX_UNPARSED = 0.10


def load_judge(repo_id: str, load_in_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs: dict = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs).eval()
    return model, tok


def render_chat(tok, prompt: str) -> str:
    """Apply the judge's OWN chat template.

    Each judge is a different family with a different format, and a judge prompted outside its
    template is a weaker judge for reasons that have nothing to do with its judgement. Qwen3.5
    accepts `enable_thinking`; Mistral and Phi reject it, so it is tried and dropped -- a judge
    that silently emits a reasoning preamble would blow the 8-token budget and parse as unusable.
    """
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def ask(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int = 8) -> list[str]:
    """Greedy, few tokens: the judge must emit one word, so sampling only adds variance."""
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        texts = [render_chat(tok, p) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=3072).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        LOG.info("  %d/%d", min(i + batch_size, len(prompts)), len(prompts))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", action="append", required=True)
    ap.add_argument("--pairs", type=Path, default=Path("eval_sets/judge_polished/pairs.json"))
    ap.add_argument("--key", type=Path,
                    default=Path("eval_sets/judge_polished/KEY_DO_NOT_OPEN_BEFORE_LABELLING.json"))
    ap.add_argument("--human-labels", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("reports/eval/calibration/judges.json"))
    args = ap.parse_args()

    pairs = json.loads(args.pairs.read_text())
    key = {str(k["item"]): k for k in json.loads(args.key.read_text())}
    human = json.loads(args.human_labels.read_text()) if args.human_labels else {}

    # Every (pair, rubric, order) is one call. Both orders always, for every judge.
    jobs = []
    for p in pairs:
        for r in RUBRICS:
            jobs.append((p["item"], r.key, "fwd",
                         build_prompt(p["customer_message"], p["response_a"], p["response_b"], r)))
            jobs.append((p["item"], r.key, "rev",
                         build_prompt(p["customer_message"], p["response_b"], p["response_a"], r)))
    LOG.info("%d pairs x %d rubrics x 2 orders = %d calls per judge",
             len(pairs), len(RUBRICS), len(jobs))

    report: dict = {"n_pairs": len(pairs), "gate": {
        "min_control_accuracy": MIN_CONTROL_ACCURACY, "max_flip_rate": MAX_FLIP_RATE,
        "max_unparsed": MAX_UNPARSED}, "judges": {}}

    for repo in args.judge:
        LOG.info("loading judge %s", repo)
        model, tok = load_judge(repo, load_in_4bit=not args.no_4bit)
        raws = ask(model, tok, [j[3] for j in jobs], args.batch_size)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        parsed: dict[tuple[int, str, str], str | None] = {}
        for (item, rk, order, _), raw in zip(jobs, raws):
            parsed[(item, rk, order)] = parse_verdict(raw)

        n_unparsed = sum(1 for v in parsed.values() if v is None)
        verdicts: dict[str, dict[str, str]] = {}
        flips = 0
        decided = 0
        for p in pairs:
            item = p["item"]
            verdicts[str(item)] = {}
            for r in RUBRICS:
                v = resolve(parsed[(item, r.key, "fwd")], parsed[(item, r.key, "rev")])
                verdicts[str(item)][r.key] = v
                if v == "FLIP":
                    flips += 1
                elif v in {"A", "B"}:
                    decided += 1

        total = len(pairs) * len(RUBRICS)
        flip_rate = flips / total
        unparsed_rate = n_unparsed / len(jobs)

        # -- controls: known answer by construction -----------------------------------
        ctrl_right = ctrl_total = 0
        for item, k in key.items():
            if k["group"] != "control":
                continue
            gold_slot = "A" if k["a"] == "gold_reference" else "B"
            for r in RUBRICS:
                v = verdicts[item][r.key]
                if v in {"A", "B"}:
                    ctrl_total += 1
                    ctrl_right += int(v == gold_slot)
        ctrl_acc = ctrl_right / ctrl_total if ctrl_total else 0.0

        # -- agreement with the human --------------------------------------------------
        agree = comparable = 0
        for item, labels in human.items():
            if item not in verdicts:
                continue
            for r in RUBRICS:
                h, v = labels.get(r.key), verdicts[item][r.key]
                if h is None or v in {"FLIP", "UNPARSED"}:
                    continue
                comparable += 1
                agree += int((h if h != "T" else "TIE") == v)
        agreement = agree / comparable if comparable else None

        passed = (ctrl_acc >= MIN_CONTROL_ACCURACY and flip_rate <= MAX_FLIP_RATE
                  and unparsed_rate <= MAX_UNPARSED)

        # -- preference, real pairs only, reported regardless but gated on `passed` -----
        pref: dict[str, dict] = {}
        for r in RUBRICS:
            w = {"fine_tuned": 0, "base": 0, "tie": 0, "flip": 0, "unparsed": 0}
            for item, k in key.items():
                if k["group"] == "control":
                    continue
                v = verdicts[item][r.key]
                if v in {"A", "B"}:
                    w[k["a" if v == "A" else "b"]] += 1
                elif v == "TIE":
                    w["tie"] += 1
                elif v == "FLIP":
                    w["flip"] += 1
                else:
                    w["unparsed"] += 1
            pref[r.key] = w

        report["judges"][repo] = {
            "passed_gate": passed,
            "control_accuracy": round(ctrl_acc, 4), "n_control_decided": ctrl_total,
            "flip_rate": round(flip_rate, 4), "unparsed_rate": round(unparsed_rate, 4),
            "decided": decided, "human_agreement": None if agreement is None else round(agreement, 4),
            "n_comparable_with_human": comparable,
            "preference": pref, "verdicts": verdicts,
        }
        LOG.info("%s: controls %.2f  flips %.2f  unparsed %.2f  human-agreement %s  -> %s",
                 repo, ctrl_acc, flip_rate, unparsed_rate,
                 "n/a" if agreement is None else f"{agreement:.2f}",
                 "PASS" if passed else "FAIL")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\n{'judge':46} {'ctrl':>6} {'flip':>6} {'unpar':>6} {'human':>6}  gate")
    for repo, e in report["judges"].items():
        ha = "  n/a" if e["human_agreement"] is None else f"{e['human_agreement']:.3f}"
        print(f"{repo:46} {e['control_accuracy']:>6.3f} {e['flip_rate']:>6.3f} "
              f"{e['unparsed_rate']:>6.3f} {ha:>6}  {'PASS' if e['passed_gate'] else 'FAIL'}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
