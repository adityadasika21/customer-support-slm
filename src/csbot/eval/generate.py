"""Generation backends for evaluation.

Two implementations behind one protocol:

``HFGenerator``      in-process transformers, optionally with a LoRA adapter attached. Used for
                     the funnel, where we evaluate a dozen checkpoints and starting a server for
                     each would be absurd.
``EndpointGenerator`` an OpenAI-compatible HTTP endpoint. Used for the headline numbers, so the
                     figures we publish describe the artefact we actually ship rather than an
                     in-notebook approximation of it.

Decoding is **greedy by default**. Sampling would add run-to-run variance that a reader cannot
distinguish from a real difference between models, and the whole point of the exercise is to make
a difference believable. Every generation run records its decoding parameters in the output file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tqdm.auto import tqdm

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeParams:
    """Decoding settings, recorded alongside every result set."""

    max_new_tokens: int = 512
    """Generous on purpose. The longest reference responses run to roughly 620 tokens, and a tight
    cap would show up as a truncation penalty that falls unevenly -- base models are typically
    more verbose than fine-tuned ones, so a low cap would flatter the fine-tune for the wrong
    reason."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 17

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    def as_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "decoding": "greedy" if self.greedy else "sampling",
        }


@runtime_checkable
class Generator(Protocol):
    """Anything that can turn rendered prompts into completions."""

    name: str

    def generate(self, prompts: list[str], params: DecodeParams) -> list[str]: ...


@dataclass
class HFGenerator:
    """In-process generation with transformers, with an optional PEFT adapter."""

    name: str
    model: object
    tokenizer: object
    batch_size: int = 8
    _warned_pad: bool = field(default=False, repr=False)

    @classmethod
    def load(
        cls,
        repo_id: str,
        *,
        name: str,
        adapter_dir: str | None = None,
        load_in_4bit: bool = False,
        batch_size: int = 8,
    ) -> HFGenerator:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(adapter_dir or repo_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Left padding is required for batched generation: with right padding the pad tokens sit
        # between the prompt and the first generated token and the model continues from padding.
        tok.padding_side = "left"

        kwargs: dict = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
            LOG.info("attached adapter %s", adapter_dir)

        model.eval()
        model.config.use_cache = True
        return cls(name=name, model=model, tokenizer=tok, batch_size=batch_size)

    def generate(self, prompts: list[str], params: DecodeParams) -> list[str]:
        import torch

        outputs: list[str] = []
        for start in tqdm(
            range(0, len(prompts), self.batch_size),
            desc=f"generate[{self.name}]",
            unit="batch",
        ):
            batch = prompts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)

            gen_kwargs = {
                "max_new_tokens": params.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": not params.greedy,
            }
            if not params.greedy:
                gen_kwargs |= {"temperature": params.temperature, "top_p": params.top_p}

            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)

            prompt_len = enc.input_ids.shape[1]
            for row in out:
                outputs.append(
                    self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
                )
        return outputs

    def unload(self) -> None:
        """Release VRAM. The funnel loads many models in sequence on one 8 GB card."""
        import gc

        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class EndpointGenerator:
    """Generation against an OpenAI-compatible HTTP server (vLLM, or our FastAPI wrapper).

    Uses the ``/v1/completions`` endpoint with an already-rendered prompt rather than
    ``/v1/chat/completions``, so the template applied is provably the one from
    ``csbot.serve.template`` and not whatever the server's own chat template does. Any drift
    between the two would silently change what is being measured.
    """

    name: str
    base_url: str
    model: str
    timeout: float = 120.0
    concurrency: int = 8

    def generate(self, prompts: list[str], params: DecodeParams) -> list[str]:
        import asyncio

        return asyncio.run(self._generate_async(prompts, params))

    async def _generate_async(self, prompts: list[str], params: DecodeParams) -> list[str]:
        import asyncio

        import httpx

        results: list[str | None] = [None] * len(prompts)
        sem = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async def one(i: int, prompt: str) -> None:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": params.max_new_tokens,
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                }
                async with sem:
                    resp = await client.post(
                        f"{self.base_url.rstrip('/')}/v1/completions", json=payload
                    )
                    resp.raise_for_status()
                    results[i] = resp.json()["choices"][0]["text"].strip()

            tasks = [one(i, p) for i, p in enumerate(prompts)]
            for fut in tqdm(
                asyncio.as_completed(tasks), total=len(tasks),
                desc=f"generate[{self.name}]", unit="req",
            ):
                await fut

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"{len(missing)} requests returned no completion")
        return [r for r in results if r is not None]
