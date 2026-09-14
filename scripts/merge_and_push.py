#!/usr/bin/env python
"""Merge a LoRA adapter into the base weights and optionally publish to the Hugging Face Hub.

Merging matters for serving. vLLM can apply LoRA at runtime, but merged weights take the
plain fast path with no per-request adapter overhead, and they let the model be loaded by anyone
with stock ``transformers`` and no PEFT dependency. Since anyone loading and running this
model against their own evaluation set, the fewer moving parts in that path the better.

The adapter is published *as well as* the merged model: it is ~50 MB against several GB, and it
is the honest artefact -- it records exactly what training changed.

    python scripts/merge_and_push.py --run artifacts/runs/final --push drstupidity/csbot-support
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

from csbot.models.registry import get as get_spec
from csbot.serve.template import SYSTEM_PROMPT

LOG = logging.getLogger("merge")


def merge_adapter(base_repo: str, adapter_dir: Path, out_dir: Path) -> Path:
    """Merge LoRA weights into the base model and save a standalone checkpoint.

    Merging happens in bf16 on CPU. A 4-bit base cannot be merged into faithfully -- dequantising
    then re-merging would bake quantisation error into the weights -- so the merge always loads
    the full-precision base regardless of how the adapter was trained.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOG.info("loading base %s in bf16 on CPU", base_repo)
    model = AutoModelForCausalLM.from_pretrained(base_repo, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter_dir), device_map="cpu")

    LOG.info("merging adapter")
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.save_pretrained(str(out_dir))
    LOG.info("merged model written to %s", out_dir)
    return out_dir


def write_model_card(run_dir: Path, out_dir: Path, spec, repo_id: str | None) -> None:
    """Model card carrying the exact prompt template.

    The model must be easy to load and run *with the documented template*, so the
    template travels with the weights rather than living only in this repo's README.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]

    card = f"""---
license: {spec.license_id}
base_model: {spec.repo_id}
library_name: transformers
tags: [customer-support, lora, fine-tuned]
---

# Customer Support Assistant ({spec.key})

LoRA fine-tune of [`{spec.repo_id}`]({f"https://huggingface.co/{spec.repo_id}"}) on the
[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
trained to answer customer-support requests in a helpful support register.

## Prompt template

Training, evaluation and serving all use this exact format. Use the model's own chat template
with `add_generation_prompt=True`{", and `enable_thinking=False`" if spec.thinking_model else ""}.

System prompt:

```
{SYSTEM_PROMPT}
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "{repo_id or str(out_dir)}"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="bfloat16", device_map="auto")

messages = [
    {{"role": "system", "content": SYSTEM_PROMPT}},
    {{"role": "user", "content": "I need to cancel order 12345"}},
]
prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True{", enable_thinking=False" if spec.thinking_model else ""}
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The model was trained with a 15% system-prompt dropout, so it also behaves sensibly with a
different system prompt or none at all.

## These weights are the model, not the served stack

Everything above is all you need. The repository also runs a guardrail layer in front of the model
at serve time (NeMo Guardrails plus a local out-of-domain embedding check), but **none of it is
baked into these weights and none of it is required to run them**. Loading the checkpoint as shown
gives you the model alone.

That separation is deliberate. The rails divert a small fraction of requests to predefined text
instead of generating -- measured at 0.27% of real validation queries, and 9.4% on a deliberately
adversarial probe set. If those fired during someone else's evaluation, the scores would describe
the rails rather than the model. So the serving layer exposes an ungated `/v1/completions` route
for exactly that purpose, and the rails can be turned off entirely with `CSBOT_GUARDRAILS=0`.

## Training

| | |
|---|---|
| method | LoRA (r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}, dropout={cfg['lora']['dropout']}) |
| target modules | {", ".join(cfg['lora']['target_modules'])} |
| learning rate | {cfg['learning_rate']} |
| schedule | {cfg['lr_scheduler_type']}, warmup {cfg['warmup_ratio']} |
| effective batch | {manifest['results']['effective_batch_size']} |
| max seq len | {cfg['data']['max_seq_len']} |
| train rows | {manifest['data']['train_rows']:,} |
| guardrail rows | {cfg['data'].get('guardrail_examples', 0):,} synthetic "when not to help" examples |
| loss | completion-only (assistant tokens) |
| hardware | {manifest['environment']['gpu']}, peak {manifest['environment']['peak_vram_gb']} GB |
| final eval loss | {manifest['results']['eval_loss']:.4f} |

Data was split by **near-duplicate cluster**, not by row: the source dataset is 27 intents with
roughly a thousand generated paraphrases each, so a random split leaks near-identical text across
the train/test boundary. See the [project repository](https://github.com/adityadasika21/customer-support-slm) for the leakage audit,
evaluation design and base-vs-tuned results.

## Why there are synthetic "decline" examples in the training data

Every one of the 23,453 Bitext rows is "customer asks -> agent helpfully assists". There is not a
single example of declining. Fine-tuning on that alone taught the model *always help*, which is
right inside the domain and wrong the moment a query leaves it: measured against a held-out
behavioural probe set, its off-topic redirect rate dropped from 0.800 (base model) to 0.350.

Blending in a 5% slice of examples that decline off-topic requests, refuse instruction overrides,
ask for clarification when a message is uninformative, and decline to invent account data brought
that to 0.850 -- past the base model -- at no measurable in-domain cost (validation loss 0.6461 vs
0.6436 for an otherwise identical control run).

## Limitations

- Trained on synthetic, English, single-turn support data. It has no multi-turn conversation
  training and no access to real account systems.
- It must not be relied on for account-specific facts; it is trained never to invent order
  numbers, policies or contact details, but that is a tendency, not a guarantee.
- Non-English input is out of distribution for the training data.
"""
    (out_dir / "README.md").write_text(card)
    LOG.info("wrote model card")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Run directory containing adapter/.")
    p.add_argument("--out", type=Path, default=None, help="Where to write merged weights.")
    p.add_argument("--push", default=None, help="HF repo id to publish to, e.g. user/model.")
    p.add_argument("--push-adapter", default=None, help="Separate repo id for the adapter alone.")
    p.add_argument("--private", action="store_true")
    p.add_argument("--skip-merge", action="store_true", help="Publish the adapter only.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    manifest = json.loads((args.run / "manifest.json").read_text())
    spec = get_spec(manifest["model"]["key"])
    adapter_dir = args.run / "adapter"
    out_dir = args.out or (args.run / "merged")

    if not args.skip_merge:
        merge_adapter(spec.repo_id, adapter_dir, out_dir)
        write_model_card(args.run, out_dir, spec, args.push)
        shutil.copy(args.run / "manifest.json", out_dir / "training_manifest.json")

    if args.push and not args.skip_merge:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.push, repo_type="model")
        LOG.info("pushed merged model to https://huggingface.co/%s", args.push)

    if args.push_adapter:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_adapter, exist_ok=True, private=args.private)
        write_model_card(args.run, adapter_dir, spec, args.push_adapter)
        api.upload_folder(folder_path=str(adapter_dir), repo_id=args.push_adapter, repo_type="model")
        LOG.info("pushed adapter to https://huggingface.co/%s", args.push_adapter)

    print(f"done. merged={out_dir if not args.skip_merge else 'skipped'}")


if __name__ == "__main__":
    main()
