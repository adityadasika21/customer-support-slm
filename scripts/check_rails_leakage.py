#!/usr/bin/env python
"""Fail if anything the guardrail layer is built from leaks into the probe set it is judged on.

The guardrail work has three sources of text that all cover the same behaviours as
``eval_sets/ood_queries.jsonl``:

* guardrail **training** examples (``csbot.data.guardrails``) -- leakage here means the model
  memorised the probe rather than learned the behaviour;
* Colang **canonical forms** (``guardrails/rails.co``) -- leakage here means the rail was seeded
  from the test set, so its measured recall is meaningless;
* in-domain **anchors** (``guardrails/anchors.json``) -- a probe query sitting among the anchors
  would be scored as in-domain by construction.

Comments claiming disjointness are not enough. On its first run this check found a real one: the
canonical form ``"tell me a joke"`` was probe ``ood-054`` verbatim, in a file whose header
asserted that no example came from the probe set.

Exits non-zero on any exact or near (Jaccard >= 0.6) collision, so it can gate a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csbot.data.guardrails import _norm, build_guardrails  # noqa: E402

NEAR_THRESHOLD = 0.6


def collisions(probes: list[str], pool: list[str], threshold: float) -> list[dict]:
    pool_tokens = [(c, set(_norm(c).split())) for c in pool]
    pool_norm = {_norm(c) for c in pool}
    out = []
    for q in probes:
        n = _norm(q)
        if n in pool_norm:
            out.append({"kind": "exact", "probe": q, "source": n, "jaccard": 1.0})
            continue
        t = set(n.split())
        if not t:
            continue
        for c, tc in pool_tokens:
            if not tc:
                continue
            j = len(t & tc) / len(t | tc)
            if j >= threshold:
                out.append({"kind": "near", "probe": q, "source": c, "jaccard": round(j, 3)})
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", type=Path, default=Path("eval_sets/ood_queries.jsonl"))
    ap.add_argument("--rails", type=Path, default=Path("guardrails"))
    ap.add_argument("--threshold", type=float, default=NEAR_THRESHOLD)
    args = ap.parse_args()

    probes = [
        json.loads(line)["query"]
        for line in args.probes.read_text().splitlines()
        if line.strip()
    ]

    sources: dict[str, list[str]] = {
        # Generated at a deliberately high count so the check covers every query the builder can
        # emit, not only the subset a particular training run happened to sample.
        "guardrail training examples": [g.instruction for g in build_guardrails(400, seed=17)],
    }

    rails_co = args.rails / "rails.co"
    if rails_co.exists():
        from nemoguardrails import RailsConfig

        cfg = RailsConfig.from_path(str(args.rails))
        sources["Colang canonical forms"] = [
            e for v in (cfg.user_messages or {}).values() for e in v
        ]

    anchors = args.rails / "anchors.json"
    if anchors.exists():
        sources["in-domain anchors"] = json.loads(anchors.read_text())["anchors"]

    failed = 0
    for name, pool in sources.items():
        hits = collisions(probes, pool, args.threshold)
        status = "FAIL" if hits else "ok"
        print(f"[{status}] {name:26} {len(pool):>5} items vs {len(probes)} probes -> {len(hits)} collisions")
        for h in hits[:10]:
            print(f"        {h['kind']:5} j={h['jaccard']:<5} {h['probe'][:58]!r}")
            print(f"              matched {h['source'][:58]!r}")
        failed += len(hits)

    print(f"\n{'LEAKAGE DETECTED' if failed else 'clean: no probe query leaks into any guardrail source'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
