"""LoRA supervised fine-tuning.

One trainer serves both the funnel pilots and the final run; the only difference is the config.
That matters for the funnel's validity -- if pilots ran through a different code path than the
final training, the pilot ranking would not transfer.

Why LoRA rather than full fine-tuning
-------------------------------------
Full fine-tuning is not physically available on 8 GB: a 1.7B model in bf16 needs weights (3.4 GB)
plus gradients (3.4 GB) plus fp32 Adam moments (~13.6 GB), over 20 GB in total.

It is also the wrong tool here, and that is the argument worth making independently of the
hardware. We are teaching style, format and domain convention -- the base instruct models already
know what a refund is. Full fine-tuning on 27 narrow intent templates invites catastrophic
forgetting, and because the model is judged on data it has never seen, forgetting is a
direct threat to the score. LoRA constrains the update to a low-rank subspace and preserves base
capability close to by construction.

Loss is computed on the assistant turn only (TRL's completion-only masking over prompt/completion
pairs). Training on the prompt as well would spend scarce adapter capacity learning to generate
*customer questions*, which the model is never asked to do.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from csbot.config import TrainConfig, git_sha
from csbot.data.loader import (
    add_guardrails,
    load_dataset_frame,
    select_rows,
    template_options,
    to_prompt_completion,
    upsample_substantive,
)
from csbot.models.registry import ModelSpec, get as get_spec

LOG = logging.getLogger(__name__)


def _supported_kwargs(cls, kwargs: dict) -> dict:
    """Filter kwargs to those the installed class actually accepts.

    TRL's config surface moves between releases. Rather than pin to one version and break on the
    next, we pass what is supported and log what was dropped, so a silent behaviour change is
    visible in the log instead of invisible in the results.
    """
    try:
        allowed = set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        return kwargs
    kept = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = set(kwargs) - set(kept)
    if dropped:
        LOG.warning("%s does not accept %s -- dropped", cls.__name__, sorted(dropped))
    return kept


def build_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=False)
    if tok.pad_token is None:
        # Padding with EOS is standard for decoder-only models; the attention mask keeps pad
        # positions out of the loss, so this does not teach the model to emit EOS spuriously.
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # left padding is for generation, not training
    return tok


def build_model(spec: ModelSpec, cfg: TrainConfig):
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    kwargs: dict = {"dtype": dtype, "device_map": {"": 0}}

    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        # NF4 with double quantisation: the information-theoretically motivated 4-bit format from
        # the QLoRA paper, with the quantisation constants themselves quantised. Compute still
        # happens in bf16, so only storage is 4-bit.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(spec.repo_id, **kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    return model


def build_peft_config(cfg: TrainConfig):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def make_sample_callback(tokenizer, prompts: list[str], out_dir: Path):
    """Generate a fixed set of held-out prompts at each evaluation and write them to disk.

    A validation-loss curve cannot distinguish "learned the support register" from "learned to
    emit the most common template". Reading a handful of actual generations every few hundred
    steps catches format collapse, repetition and truncation while there is still time to react.

    Built as a closure so it binds to whatever ``TrainerCallback`` the installed transformers
    provides.
    """
    from transformers import TrainerCallback

    out_dir.mkdir(parents=True, exist_ok=True)

    class _Callback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if model is None or not prompts:
                return
            was_training = model.training
            model.eval()
            use_cache = model.config.use_cache
            model.config.use_cache = True
            records = []
            try:
                with torch.no_grad():
                    for prompt in prompts:
                        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
                        out = model.generate(
                            **batch,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        text = tokenizer.decode(
                            out[0][batch.input_ids.shape[1] :], skip_special_tokens=True
                        )
                        records.append({"prompt": prompt, "generation": text})
            except Exception as exc:  # never let sampling kill a training run
                LOG.warning("sample generation failed at step %s: %s", state.global_step, exc)
            finally:
                model.config.use_cache = use_cache
                if was_training:
                    model.train()

            path = out_dir / f"samples_step{state.global_step:06d}.json"
            path.write_text(json.dumps(records, indent=2))
            LOG.info("wrote %d sample generations to %s", len(records), path)

    return _Callback()


@dataclass
class TrainResult:
    run_dir: Path
    adapter_dir: Path
    metrics: dict
    manifest: dict


def train(cfg: TrainConfig, resume_from: str | Path | None = None) -> TrainResult:
    """Run one fine-tuning job and return its artefacts."""
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)
    spec = get_spec(cfg.model_key)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("run=%s model=%s variant=%s scope=%s",
             cfg.run_name, spec.repo_id, cfg.data.variant, cfg.data.scope)

    tokenizer = build_tokenizer(spec)
    frame = load_dataset_frame(cfg.data.parquet)

    train_rows = select_rows(
        frame, slice_name="train", scope=cfg.data.scope, variant=cfg.data.variant
    )
    val_rows = select_rows(
        frame, slice_name="val", scope=cfg.data.scope, variant=cfg.data.variant
    )
    if cfg.data.max_train_rows:
        train_rows = train_rows.sample(
            n=min(cfg.data.max_train_rows, len(train_rows)), random_state=cfg.seed
        ).reset_index(drop=True)
    # Blended AFTER any cap, so the guardrail proportion is what the config asks for rather than
    # being diluted or erased by sampling.
    train_rows = upsample_substantive(
        train_rows, cfg.data.substantive_upsample, seed=cfg.seed
    )
    train_rows = add_guardrails(train_rows, cfg.data.guardrail_examples, seed=cfg.seed)
    train_rows = train_rows.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    val_rows = val_rows.head(cfg.eval_rows)

    train_ds = to_prompt_completion(train_rows, tokenizer, spec, training=True, seed=cfg.seed)
    val_ds = to_prompt_completion(val_rows, tokenizer, spec, training=False, seed=cfg.seed)
    LOG.info("train=%d val=%d", len(train_ds), len(val_ds))

    model = build_model(spec, cfg)

    sft_kwargs = dict(
        output_dir=str(run_dir / "checkpoints"),
        run_name=cfg.run_name,
        seed=cfg.seed,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        # Three, not one. With a single kept checkpoint a crash at step 590 of 600 rewinds to
        # the last save and there is nothing older to fall back to if that save was mid-write.
        # Three costs ~500 MB of adapter checkpoints and buys a real recovery window.
        report_to=[],
        max_length=cfg.data.max_seq_len,
        completion_only_loss=True,
        packing=False,  # packing would blur the completion-only mask across examples
        dataset_num_proc=1,
    )
    args = SFTConfig(**_supported_kwargs(SFTConfig, sft_kwargs))

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg),
    )
    trainer = SFTTrainer(**_supported_kwargs(SFTTrainer, trainer_kwargs))

    if cfg.sample_generations:
        from csbot.serve.template import render_prompt

        options = template_options(spec)
        prompts = [
            render_prompt(tokenizer, r.instruction, options=options)
            for r in val_rows.head(cfg.sample_generations).itertuples(index=False)
        ]
        trainer.add_callback(make_sample_callback(tokenizer, prompts, run_dir / "samples"))

    started = time.time()
    # Resume from the newest checkpoint in this run's directory when asked. Without this a
    # power cut or a crash discards every step since the last save AND cannot pick them up --
    # the checkpoint is written but nothing ever reads it back.
    resume = None
    if resume_from:
        ckpt_dir = run_dir / "checkpoints"
        if str(resume_from) == "auto":
            found = sorted(ckpt_dir.glob("checkpoint-*"),
                           key=lambda q: int(q.name.split("-")[-1]))
            resume = str(found[-1]) if found else None
            if resume is None:
                LOG.warning("no checkpoint under %s; starting from scratch", ckpt_dir)
        else:
            resume = str(resume_from)
        if resume:
            LOG.info("resuming from %s", resume)

    train_out = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - started

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    eval_metrics = trainer.evaluate()
    history = trainer.state.log_history
    (run_dir / "log_history.json").write_text(json.dumps(history, indent=2))

    manifest = {
        "run_name": cfg.run_name,
        "git_sha": git_sha(),
        "model": {
            "key": spec.key, "repo_id": spec.repo_id,
            "params_b": spec.params_b, "license": spec.license_id,
        },
        "config": cfg.to_dict(),
        "data": {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "variant": cfg.data.variant,
            "scope": cfg.data.scope,
        },
        "results": {
            "train_runtime_s": round(elapsed, 1),
            "train_loss": train_out.metrics.get("train_loss"),
            "eval_loss": eval_metrics.get("eval_loss"),
            "steps": trainer.state.global_step,
            "effective_batch_size": cfg.effective_batch_size,
        },
        "environment": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available() else None,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("done in %.1fs -- eval_loss=%s peak_vram=%sGB",
             elapsed, eval_metrics.get("eval_loss"),
             manifest["environment"]["peak_vram_gb"])

    return TrainResult(run_dir, adapter_dir, eval_metrics, manifest)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="LoRA fine-tune a candidate model.")
    p.add_argument("config", type=Path)
    p.add_argument("--run-name", default=None)
    p.add_argument("--model-key", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--variant", default=None,
                   choices=["raw", "substituted", "grounded", "polished"])
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="Resume training. Bare --resume picks the newest checkpoint in the "
                        "run's own directory; pass a path to choose one.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from csbot.config import load_train_config

    overrides = {k: v for k, v in
                 {"run_name": args.run_name, "model_key": args.model_key,
                  "max_steps": args.max_steps}.items() if v is not None}
    cfg = load_train_config(args.config, **overrides)
    if args.variant:
        cfg.data.variant = args.variant

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
