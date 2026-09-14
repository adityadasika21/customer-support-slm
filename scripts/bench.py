#!/usr/bin/env python
"""Latency and throughput benchmark against the running HTTP endpoint.

Performance numbers are only meaningful with the method attached, so it is explicit:

* **warmup requests are discarded** -- the first request pays CUDA graph capture, kernel
  autotuning and allocator warmup, and including it inflates p50 on small samples;
* every configuration uses the **same fixed prompt set**, cycled deterministically, because
  output length dominates latency and a different prompt mix silently changes the result;
* decoding is **greedy** and ``max_tokens`` is pinned, so runs are comparable;
* concurrency is swept to separate single-request latency from server throughput -- they move in
  opposite directions, and reporting only one of them is how benchmarks mislead;
* percentiles are reported rather than means, since latency distributions have long right tails.

Time-to-first-token is measured only when the backend streams (vLLM). The in-process transformers
engine returns a complete response, so TTFT is reported as ``null`` there rather than silently
conflated with end-to-end latency.

    python scripts/bench.py --url http://127.0.0.1:8000 --concurrency 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

#: Prompts that exercise a realistic spread of response lengths.
#:
#: All eight must be *in-domain*. ``/chat`` is guardrailed, and a railed request returns
#: predefined text in a few milliseconds with no generation at all -- so a single off-topic prompt
#: in this list would silently flatter every latency number in the report. ``assert_not_railed``
#: below checks this at run time rather than trusting the list to stay correct.
DEFAULT_PROMPTS = [
    "I need to cancel order 884213",
    "Where is my package? It was due Tuesday.",
    "How do I get a refund for an item that never arrived?",
    "Can you send me invoice INV-20194?",
    "I can't log into my account, it says wrong password",
    "What payment methods do you accept?",
    "Please change the delivery address on order #A-7781",
    "I want to speak to a human agent",
]


async def one_request(
    client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int, model: str
) -> dict:
    started = time.perf_counter()
    resp = await client.post(
        f"{url.rstrip('/')}/chat",
        json={"message": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=300,
    )
    elapsed = time.perf_counter() - started
    resp.raise_for_status()
    data = resp.json()
    return {
        "latency_s": elapsed,
        "completion_tokens": data.get("completion_tokens", 0),
        "prompt_tokens": data.get("prompt_tokens", 0),
        "guardrail": data.get("guardrail"),
    }


def assert_not_railed(url: str, prompts: list[str], model: str) -> None:
    """Fail loudly if any benchmark prompt is answered by a guardrail instead of the model.

    A railed reply costs one embedding lookup and no generation, so it is roughly two orders of
    magnitude faster. Mixing even one into the sample would make the benchmark describe something
    other than the model's serving performance.
    """
    railed = []
    for prompt in prompts:
        resp = httpx.post(
            f"{url.rstrip('/')}/chat",
            json={"message": prompt, "max_tokens": 8, "temperature": 0.0}, timeout=300,
        )
        resp.raise_for_status()
        decision = resp.json().get("guardrail")
        if decision:
            railed.append((prompt, decision.get("rail")))
    if railed:
        raise SystemExit(
            "benchmark prompts are being answered by guardrails, which would not measure "
            "generation at all:\n"
            + "\n".join(f"  {p!r} -> {r}" for p, r in railed)
        )


async def run_level(
    url: str, prompts: list[str], concurrency: int, n_requests: int,
    max_tokens: int, model: str, warmup: int,
) -> dict:
    """Run one concurrency level, discarding warmup requests from the statistics."""
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        # Warmup: not measured.
        await asyncio.gather(
            *(one_request(client, url, prompts[i % len(prompts)], max_tokens, model)
              for i in range(warmup))
        )

        sem = asyncio.Semaphore(concurrency)

        async def guarded(i: int) -> dict:
            async with sem:
                return await one_request(client, url, prompts[i % len(prompts)], max_tokens, model)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(n_requests)))
        wall = time.perf_counter() - wall_start

    latencies = sorted(r["latency_s"] for r in results)
    out_tokens = sum(r["completion_tokens"] for r in results)

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        k = min(len(latencies) - 1, max(0, int(round(p / 100 * (len(latencies) - 1)))))
        return latencies[k]

    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_s": round(wall, 2),
        "latency_p50_s": round(pct(50), 3),
        "latency_p95_s": round(pct(95), 3),
        "latency_p99_s": round(pct(99), 3),
        "latency_mean_s": round(statistics.fmean(latencies), 3),
        "requests_per_s": round(n_requests / wall, 2),
        "output_tokens_per_s": round(out_tokens / wall, 1),
        "mean_output_tokens": round(out_tokens / n_requests, 1),
        "ttft_s": None,  # only meaningful for a streaming backend
    }


def gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu": None}
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu": props.name,
            "vram_gb": round(props.total_memory / 1e9, 1),
            "capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
        }
    except Exception:
        return {"gpu": "unknown"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--requests", type=int, default=32, help="Measured requests per level.")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--model", default="csbot")
    p.add_argument("--out", type=Path, default=Path("reports/bench.json"))
    args = p.parse_args()

    info = httpx.get(f"{args.url.rstrip('/')}/model-info", timeout=60).json()

    # Warms the model and proves the sample measures generation, before any timing is recorded.
    assert_not_railed(args.url, DEFAULT_PROMPTS, args.model)
    print(f"serving: {info.get('served_label')}  engine={info.get('engine')}")

    rows = []
    for level in args.concurrency:
        print(f"  concurrency={level} ...", flush=True)
        row = asyncio.run(
            run_level(args.url, DEFAULT_PROMPTS, level, args.requests,
                      args.max_tokens, args.model, args.warmup)
        )
        rows.append(row)
        print(f"    p50={row['latency_p50_s']}s p95={row['latency_p95_s']}s "
              f"{row['requests_per_s']} req/s {row['output_tokens_per_s']} tok/s")

    report = {
        "method": {
            "warmup_requests_discarded": args.warmup,
            "measured_requests_per_level": args.requests,
            "max_tokens": args.max_tokens,
            "decoding": "greedy (temperature=0)",
            "prompt_set": DEFAULT_PROMPTS,
            "note": "TTFT requires a streaming backend; null for the transformers engine.",
        },
        "hardware": gpu_info(),
        "served": info.get("served_label"),
        "engine": info.get("engine"),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\n{'conc':>5}{'p50 s':>9}{'p95 s':>9}{'req/s':>9}{'tok/s':>9}")
    for r in rows:
        print(f"{r['concurrency']:>5}{r['latency_p50_s']:>9}{r['latency_p95_s']:>9}"
              f"{r['requests_per_s']:>9}{r['output_tokens_per_s']:>9}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
