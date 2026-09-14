"""HTTP API for the fine-tuned support assistant.

Two engines behind one interface:

``local``   transformers in-process. Always works, needs no extra daemon, and is what the
            reproduction instructions default to.
``proxy``   forwards to a vLLM OpenAI-compatible server for real throughput. vLLM on sm_120
            Blackwell is new enough to be a risk, so it is the performance path rather than the
            only path -- serving must not become the critical path.

The server exposes both a friendly ``/chat`` endpoint and an OpenAI-compatible
``/v1/completions``. The latter exists so the evaluation harness can run against the *served*
model over HTTP (see ``csbot.eval.generate.EndpointGenerator``), which means the numbers we
publish describe the artefact we actually ship rather than an in-process approximation of it.

Crucially, ``/chat`` renders prompts with ``csbot.serve.template`` -- the same module the trainer
used. Train/serve template drift is the classic silent failure here, and this is the one place it
could creep back in.

    uvicorn csbot.serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from csbot.models.registry import ModelSpec, get as get_spec
from csbot.serve.guardrails import GuardrailLayer
from csbot.serve.template import (
    SYSTEM_PROMPT,
    TemplateOptions,
    render_prompt,
)

LOG = logging.getLogger("csbot.serve")

# -- configuration via environment, so the same image serves any run -------------------------
MODEL_KEY = os.environ.get("CSBOT_MODEL_KEY", "qwen3-1.7b")
ADAPTER_DIR = os.environ.get("CSBOT_ADAPTER_DIR") or None
MERGED_DIR = os.environ.get("CSBOT_MERGED_DIR") or None
ENGINE_MODE = os.environ.get("CSBOT_ENGINE", "local")
PROXY_URL = os.environ.get("CSBOT_PROXY_URL", "http://127.0.0.1:8001")
PROXY_MODEL = os.environ.get("CSBOT_PROXY_MODEL") or None
"""Which served model to proxy to.

Necessary once the backend serves more than one: vLLM can hold the base weights plus several
LoRA adapters at once, and taking `data[0]` then silently proxies to whichever the server happens
to list first -- quite possibly the *base* model, which would make every served answer, and every
benchmark, describe the wrong thing."""
LOAD_IN_4BIT = os.environ.get("CSBOT_LOAD_IN_4BIT", "0") == "1"
GUARDRAILS = os.environ.get("CSBOT_GUARDRAILS", "1") == "1"
GUARDRAILS_CONFIG = os.environ.get("CSBOT_GUARDRAILS_CONFIG", "guardrails")

DEFAULT_MAX_TOKENS = 512


class ChatRequest(BaseModel):
    message: str = Field(..., description="The customer's message.")
    system: str | None = Field(
        None, description="Override the system prompt. Omit to use the trained default."
    )
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)


class ChatResponse(BaseModel):
    reply: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    guardrail: dict | None = None
    """Present when a rail answered instead of the model.

    Reported rather than hidden: a caller benchmarking the system needs to know which replies
    involved no generation, and an evaluation that silently mixed canned and generated text
    would report a model quality it never measured.
    """


class CompletionRequest(BaseModel):
    """Minimal OpenAI-compatible shape, enough for the evaluation harness."""

    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class LocalEngine:
    """In-process transformers generation."""

    spec: ModelSpec
    model: Any
    tokenizer: Any
    label: str

    @classmethod
    def load(cls) -> LocalEngine:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = get_spec(MODEL_KEY)
        source = MERGED_DIR or spec.repo_id
        tok = AutoTokenizer.from_pretrained(MERGED_DIR or ADAPTER_DIR or spec.repo_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        kwargs: dict = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        if LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        label = source
        if ADAPTER_DIR and not MERGED_DIR:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ADAPTER_DIR)
            label = f"{source}+{ADAPTER_DIR}"

        model.eval()
        model.config.use_cache = True
        LOG.info("loaded %s (4bit=%s)", label, LOAD_IN_4BIT)
        return cls(spec=spec, model=model, tokenizer=tok, label=label)

    def complete(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float) -> dict:
        import torch

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs |= {"temperature": temperature, "top_p": top_p}

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        prompt_len = enc.input_ids.shape[1]
        completion_ids = out[0][prompt_len:]
        return {
            "text": self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(completion_ids.shape[0]),
        }


@dataclass
class ProxyEngine:
    """Forwards to a vLLM OpenAI-compatible server."""

    spec: ModelSpec
    tokenizer: Any
    base_url: str
    model_name: str
    label: str

    @classmethod
    def load(cls) -> ProxyEngine:
        import httpx
        from transformers import AutoTokenizer

        spec = get_spec(MODEL_KEY)
        tok = AutoTokenizer.from_pretrained(MERGED_DIR or ADAPTER_DIR or spec.repo_id)

        # Ask the backend what it is actually serving rather than assuming; a mismatch here is
        # exactly the kind of thing that silently invalidates a benchmark.
        try:
            resp = httpx.get(f"{PROXY_URL.rstrip('/')}/v1/models", timeout=10)
            resp.raise_for_status()
            served = [m["id"] for m in resp.json()["data"]]
            if PROXY_MODEL:
                if PROXY_MODEL not in served:
                    raise RuntimeError(
                        f"CSBOT_PROXY_MODEL={PROXY_MODEL!r} is not served; backend has {served}"
                    )
                model_name = PROXY_MODEL
            elif len(served) > 1:
                raise RuntimeError(
                    f"backend serves {served}; set CSBOT_PROXY_MODEL to choose one rather than "
                    "silently proxying to whichever is listed first"
                )
            else:
                model_name = served[0]
        except Exception as exc:  # pragma: no cover - depends on a live backend
            raise RuntimeError(f"vLLM backend unreachable at {PROXY_URL}: {exc}") from exc

        LOG.info("proxying to %s serving %s", PROXY_URL, model_name)
        return cls(spec=spec, tokenizer=tok, base_url=PROXY_URL,
                   model_name=model_name, label=f"vllm:{model_name}")

    def complete(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float) -> dict:
        import httpx

        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/v1/completions",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "text": data["choices"][0]["text"].strip(),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }


STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    STATE["engine"] = (ProxyEngine if ENGINE_MODE == "proxy" else LocalEngine).load()
    STATE["options"] = TemplateOptions(
        supports_system_role=STATE["engine"].spec.supports_system_role,
        chat_template_kwargs=dict(STATE["engine"].spec.chat_template_kwargs),
    )

    rails = GuardrailLayer(GUARDRAILS_CONFIG, enabled=GUARDRAILS)
    if GUARDRAILS:
        # Build the embedding index at startup, not on the first customer request: otherwise the
        # first request pays several seconds and every latency benchmark starts with an outlier.
        await rails._build()
    STATE["guardrails"] = rails
    yield
    STATE.clear()


app = FastAPI(
    title="Customer Support Assistant",
    version="0.1.0",
    description="Fine-tuned small language model served behind an HTTP API.",
    lifespan=lifespan,
)


def _rails() -> GuardrailLayer:
    """The guardrail layer, or a disabled one.

    Falling back rather than raising keeps the template and usage tests able to drive the app
    with a hand-built STATE, and means a guardrail misconfiguration degrades to "no rails"
    instead of taking the whole service down.
    """
    rails = STATE.get("guardrails")
    if rails is None:
        rails = STATE["guardrails"] = GuardrailLayer(GUARDRAILS_CONFIG, enabled=False)
    return rails


def _engine():
    engine = STATE.get("engine")
    if engine is None:  # pragma: no cover - only before startup completes
        raise HTTPException(503, "engine not ready")
    return engine


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if STATE.get("engine") else "loading", "engine": ENGINE_MODE}


@app.get("/model-info")
async def model_info() -> dict:
    """Everything needed to reproduce a request exactly, including the prompt template."""
    engine = _engine()
    return {
        "model_key": engine.spec.key,
        "base_repo_id": engine.spec.repo_id,
        "license": engine.spec.license_id,
        "params_b": engine.spec.params_b,
        "adapter_dir": ADAPTER_DIR,
        "merged_dir": MERGED_DIR,
        "engine": ENGINE_MODE,
        "served_label": engine.label,
        "system_prompt": SYSTEM_PROMPT,
        "chat_template_kwargs": dict(engine.spec.chat_template_kwargs),
        "example_rendered_prompt": render_prompt(
            engine.tokenizer, "I need to cancel order 12345", options=STATE["options"]
        ),
        "defaults": {"max_tokens": DEFAULT_MAX_TOKENS, "temperature": 0.0},
        "guardrails": await _rails().describe(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Answer one customer message using the exact trained prompt template.

    Guardrails run first. A request stopped by a rail is answered from reviewable predefined
    text and never reaches the model -- see ``csbot.serve.guardrails`` for why NeMo is used for
    policy but never for generation.
    """
    engine = _engine()
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")

    started = time.perf_counter()
    rails = _rails()
    decision = await rails.check(req.message)
    if decision is not None:
        return ChatResponse(
            reply=decision.message,
            model=f"{engine.label} (guardrail)",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            guardrail={"rail": decision.rail, "stage": decision.stage,
                       "score": decision.score, "generated": False},
        )

    prompt = render_prompt(
        engine.tokenizer,
        req.message,
        system=req.system if req.system is not None else SYSTEM_PROMPT,
        options=STATE["options"],
    )

    # Off the event loop. `/chat` is async (the guardrail check is), but `engine.complete` is a
    # blocking HTTP call -- awaiting nothing while it runs serialises every concurrent request on
    # the single loop thread. Measured before this fix: throughput pinned at 40 tok/s with p50
    # rising linearly from 4.6 s at concurrency 1 to 46.6 s at 24, while vLLM itself sustained
    # 957 tok/s. The server was the bottleneck, not the model.
    result = await asyncio.to_thread(
        engine.complete, prompt,
        max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    return ChatResponse(
        reply=result["text"],
        model=engine.label,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        latency_ms=round(latency_ms, 1),
    )


@app.post("/v1/completions")
def completions(req: CompletionRequest) -> dict:
    """OpenAI-compatible raw-prompt completion.

    Takes an already-rendered prompt and applies no template of its own, so the evaluation
    harness controls the exact bytes the model sees and cannot be silently second-guessed by a
    server-side chat template.
    """
    engine = _engine()
    prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)

    choices, prompt_tokens, completion_tokens = [], 0, 0
    for i, prompt in enumerate(prompts):
        result = engine.complete(
            prompt, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p,
        )
        choices.append({"index": i, "text": result["text"], "finish_reason": "stop"})
        prompt_tokens += result["prompt_tokens"]
        completion_tokens += result["completion_tokens"]

    return {
        "object": "text_completion",
        "model": engine.label,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
