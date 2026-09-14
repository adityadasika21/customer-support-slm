#!/usr/bin/env python
"""Compare arms on the behavioural probes, isolating what each mitigation actually bought.

The over-compliance regression (WORKLOG §30) has two independent mitigations that fail in
different ways, so they are measured separately rather than shipped together and credited jointly:

* **data** -- guardrail examples blended into training. No runtime cost, but probabilistic: it
  shifts a tendency, it cannot enforce a rule.
* **rails** -- the NeMo policy layer. Deterministic and auditable, but only acts on what it can
  classify, and costs an embedding lookup per request.

Any arm can be run with ``+rails`` appended, which applies the guardrail layer in front of that
arm exactly as the server does: a railed request is answered from predefined text and never
reaches the model. So the four-way grid (control / data / control+rails / data+rails) separates
the two contributions and shows whether they overlap.

Every arm answers byte-identical rendered prompts under identical greedy decoding, and the
comparison is paired per probe instance -- the same query, the same probe, both arms -- so the
bootstrap interval is over the *difference*, which is far tighter than comparing two independent
rates.

    python scripts/probe_ab.py \
      --arm control=artifacts/runs/pilot-granite-3.3-2b/adapter \
      --arm data=artifacts/runs/pilot-granite-guardrail/adapter \
      --arm data+rails=artifacts/runs/pilot-granite-guardrail/adapter:rails \
      --reference control
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from csbot.data.loader import load_ood_rows
from csbot.eval.generate import DecodeParams, HFGenerator
from csbot.eval.probes import run_probes
from csbot.eval.runner import render_prompts
from csbot.models.registry import get as get_spec

LOG = logging.getLogger("probe_ab")
BOOTSTRAP = 10_000
SEED = 17


def parse_arm(spec: str) -> tuple[str, str | None, bool]:
    """``label=source[:rails]`` -> (label, source or None, rails on).

    ``source`` is a local adapter directory, or ``@name`` for a model already served by vLLM.
    The endpoint form exists so probes can run while vLLM holds the card: loading a second copy
    of the weights in-process would not fit in 8 GB alongside it.
    """
    label, _, rest = spec.partition("=")
    rails = rest.endswith(":rails")
    if rails:
        rest = rest[: -len(":rails")]
    return label, (rest or None), rails


def apply_rails(queries: list[str], config: Path) -> list[tuple[bool, str, str]]:
    """Run the serving guardrail layer over the queries.

    Returns ``(blocked, message, rail)`` per query. This is the same ``GuardrailLayer`` the API
    uses, not a reimplementation, so a divergence between measurement and production is not
    possible without the test catching it.
    """
    from csbot.serve.guardrails import GuardrailLayer

    layer = GuardrailLayer(config, enabled=True)

    async def run():
        out = []
        for q in queries:
            d = await layer.check(q)
            out.append((True, d.message, d.rail) if d else (False, "", ""))
        return out

    return asyncio.run(run())


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """CI for the mean paired difference ``b - a``, resampling probe instances."""
    diff = b.astype(float) - a.astype(float)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, help="label=adapter_path[:rails]")
    ap.add_argument("--model-key", default="granite-3.3-2b")
    ap.add_argument("--reference", default=None, help="Arm to compare the others against.")
    ap.add_argument("--rails-config", type=Path, default=Path("guardrails"))
    ap.add_argument("--probes", type=Path, default=Path("eval_sets/ood_queries.jsonl"))
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--out", type=Path, default=Path("reports/eval/probe_ab"))
    ap.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/probe_cache"),
        help="Raw generations are cached per (adapter, probe file, decode settings). Re-running "
             "only to change a rail setting then costs no GPU at all -- which matters when the "
             "GPU is busy training, and means the rails are compared against identical text "
             "rather than a fresh sample.",
    )
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001")
    ap.add_argument("--endpoint-concurrency", type=int, default=24)
    args = ap.parse_args()

    spec = get_spec(args.model_key)
    rows = load_ood_rows(args.probes)
    LOG.info("%d probe queries across %d buckets", len(rows), rows["category"].nunique())

    arms = [parse_arm(a) for a in args.arm]
    args.out.mkdir(parents=True, exist_ok=True)

    # Rail decisions depend only on the query, so compute them once and reuse across arms.
    rail_cache: list[tuple[bool, str, str]] | None = None
    if any(r for _, _, r in arms):
        LOG.info("running the guardrail layer over the probe queries")
        rail_cache = apply_rails(rows["instruction"].tolist(), args.rails_config)
        LOG.info("rails would stop %d/%d probe queries",
                 sum(1 for b, _, _ in rail_cache if b), len(rail_cache))

    # Group arms by adapter so each set of weights is loaded once, not once per rail setting.
    by_adapter: dict[str | None, list[tuple[str, bool]]] = {}
    for label, adapter, rails in arms:
        by_adapter.setdefault(adapter, []).append((label, rails))

    probe_frames: dict[str, pd.DataFrame] = {}
    generations: dict[str, list[str]] = {}

    import hashlib

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    for adapter, labels in by_adapter.items():
        # The cache key covers everything that changes the text: which weights, which queries,
        # and how many tokens. A stale hit would silently compare rails against the wrong model.
        key = hashlib.sha256(
            f"{adapter}|{args.probes.read_text()}|{args.model_key}|"
            f"{args.max_new_tokens}|{args.load_in_4bit}".encode()
        ).hexdigest()[:16]
        # An endpoint arm and an in-process arm are different execution paths and can differ
        # numerically, so they never share a cache entry.
        cache_file = args.cache_dir / f"gen-{key}.json"

        if cache_file.exists() and not args.no_cache:
            texts = json.loads(cache_file.read_text())
            LOG.info("reusing cached generations for %s (%s)", adapter or "base", cache_file.name)
        elif adapter and adapter.startswith("@"):
            from transformers import AutoTokenizer

            from csbot.eval.generate import EndpointGenerator

            served = adapter[1:]
            LOG.info("generating via endpoint %s as model %r", args.endpoint, served)
            tok = AutoTokenizer.from_pretrained(spec.repo_id)
            prompts = render_prompts(rows, tok, spec)
            gen = EndpointGenerator(name=served, base_url=args.endpoint, model=served,
                                    concurrency=args.endpoint_concurrency, timeout=600.0)
            texts = gen.generate(prompts, DecodeParams(max_new_tokens=args.max_new_tokens))
            cache_file.write_text(json.dumps(texts))
        else:
            LOG.info("loading %s", adapter or "base model (no adapter)")
            gen = HFGenerator.load(
                spec.repo_id, name=adapter or "base", adapter_dir=adapter,
                load_in_4bit=args.load_in_4bit, batch_size=args.batch_size,
            )
            prompts = render_prompts(rows, gen.tokenizer, spec)
            texts = gen.generate(prompts, DecodeParams(max_new_tokens=args.max_new_tokens))
            cache_file.write_text(json.dumps(texts))
            del gen
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        for label, rails in labels:
            if rails and rail_cache is not None:
                merged = [msg if blocked else t
                          for t, (blocked, msg, _) in zip(texts, rail_cache)]
            else:
                merged = list(texts)
            generations[label] = merged
            scored = rows.copy()
            scored["generation"] = merged
            probe_frames[label] = run_probes(scored)
            LOG.info("%-14s probe pass rate %.3f (n=%d)",
                     label, probe_frames[label]["passed"].mean(), len(probe_frames[label]))

    # ---- report ---------------------------------------------------------------------
    reference = args.reference or arms[0][0]
    rng = np.random.default_rng(SEED)
    labels = [a[0] for a in arms]

    key = ["id", "probe"]
    ref_df = probe_frames[reference].sort_values(key).reset_index(drop=True)

    summary: dict = {
        "n_probe_queries": int(len(rows)),
        "reference": reference,
        "bootstrap_samples": BOOTSTRAP,
        "arms": {},
    }
    for label in labels:
        df = probe_frames[label].sort_values(key).reset_index(drop=True)
        assert (df["id"].values == ref_df["id"].values).all(), "probe instances misaligned"
        entry = {
            "overall": round(float(df["passed"].mean()), 4),
            "n_instances": int(len(df)),
            "by_probe": {
                p: round(float(g["passed"].mean()), 4)
                for p, g in df.groupby("probe")
            },
        }
        if label != reference:
            lo, hi = paired_bootstrap(ref_df["passed"].values, df["passed"].values, rng)
            entry["delta_vs_reference"] = round(float(df["passed"].mean() - ref_df["passed"].mean()), 4)
            entry["ci95"] = [round(lo, 4), round(hi, 4)]
            entry["significant"] = bool(lo > 0 or hi < 0)
            entry["by_probe_delta"] = {
                p: round(float(g["passed"].mean() - ref_df[ref_df["probe"] == p]["passed"].mean()), 4)
                for p, g in df.groupby("probe")
            }
        summary["arms"][label] = entry
        df.to_csv(args.out / f"probes_{label.replace('+', '_')}.csv", index=False)
        pd.DataFrame({"query": rows["instruction"], "bucket": rows["category"],
                      "generation": generations[label]}).to_csv(
            args.out / f"generations_{label.replace('+', '_')}.csv", index=False)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    probes = sorted(ref_df["probe"].unique())
    width = max(len(p) for p in probes) + 2
    print(f"\n{'probe':<{width}}" + "".join(f"{l:>16}" for l in labels))
    for p in probes:
        cells = "".join(f"{summary['arms'][l]['by_probe'].get(p, float('nan')):>16.3f}" for l in labels)
        print(f"{p:<{width}}{cells}")
    print(f"{'OVERALL':<{width}}" + "".join(f"{summary['arms'][l]['overall']:>16.3f}" for l in labels))
    print(f"\nreference: {reference}")
    for label in labels:
        e = summary["arms"][label]
        if "ci95" in e:
            flag = "significant" if e["significant"] else "not significant"
            print(f"  {label:<16} delta {e['delta_vs_reference']:+.4f}  "
                  f"95% CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]  {flag}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
