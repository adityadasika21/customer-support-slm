"""Tests for the HTTP API, using a fake engine so no GPU or model download is needed.

The point of these is the contract, not the generation: that ``/chat`` applies the documented
template, that ``/v1/completions`` applies *no* template (the eval harness controls those bytes),
and that ``/model-info`` publishes enough to reproduce a request.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from csbot.models.registry import get as get_spec
from csbot.serve import app as app_module
from csbot.serve.template import SYSTEM_PROMPT, TemplateOptions


class FakeTokenizer:
    eos_token = "<|end|>"
    pad_token = "<|end|>"

    def apply_chat_template(self, conversation, *, tokenize=False, add_generation_prompt=False, **kw):
        parts = [f"<|{m['role']}|>{m['content']}<|/{m['role']}|>" for m in conversation]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        if kw.get("enable_thinking") is False:
            parts.append("<|nothink|>")
        return "".join(parts)


class FakeEngine:
    """Records the exact prompt it was handed, so tests can assert on the rendered bytes."""

    def __init__(self):
        self.spec = get_spec("qwen3-1.7b")
        self.tokenizer = FakeTokenizer()
        self.label = "fake-engine"
        self.seen: list[str] = []

    def complete(self, prompt, *, max_tokens, temperature, top_p):
        self.seen.append(prompt)
        return {"text": "Happy to help with that.", "prompt_tokens": 11, "completion_tokens": 5}


def make_client():
    """Install a fake engine and return a client.

    Deliberately bypasses the app's lifespan, which would load a real model onto the GPU.
    """
    engine = FakeEngine()
    app_module.STATE["engine"] = engine
    app_module.STATE["options"] = TemplateOptions(
        supports_system_role=engine.spec.supports_system_role,
        chat_template_kwargs=dict(engine.spec.chat_template_kwargs),
    )
    return TestClient(app_module.app), engine


def test_health_reports_ok():
    client, _ = make_client()
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_model_info_publishes_the_template():
    """Anyone must be able to reproduce the exact prompt without reading the source."""
    client, _ = make_client()
    info = client.get("/model-info").json()

    assert info["system_prompt"] == SYSTEM_PROMPT
    assert info["base_repo_id"] == "Qwen/Qwen3-1.7B"
    assert info["license"] == "apache-2.0"
    # Thinking must be disabled for a support assistant.
    assert info["chat_template_kwargs"] == {"enable_thinking": False}
    assert "<|assistant|>" in info["example_rendered_prompt"]


def test_chat_applies_the_documented_template():
    client, engine = make_client()
    resp = client.post("/chat", json={"message": "I need to cancel order 12345"})
    assert resp.status_code == 200

    sent = engine.seen[-1]
    assert SYSTEM_PROMPT in sent
    assert "I need to cancel order 12345" in sent
    assert sent.endswith("<|nothink|>")  # generation prompt + thinking disabled


def test_chat_allows_system_override():
    """A caller may supply their own system prompt; the endpoint must honour it."""
    client, engine = make_client()
    client.post("/chat", json={"message": "hi", "system": "You are a pirate."})
    assert "You are a pirate." in engine.seen[-1]
    assert SYSTEM_PROMPT not in engine.seen[-1]


def test_chat_returns_usage_and_latency():
    client, _ = make_client()
    body = client.post("/chat", json={"message": "hello"}).json()
    assert body["reply"] == "Happy to help with that."
    assert body["completion_tokens"] == 5
    assert body["latency_ms"] >= 0


def test_empty_message_rejected():
    client, _ = make_client()
    assert client.post("/chat", json={"message": "   "}).status_code == 400


def test_completions_applies_no_template():
    """The eval harness sends pre-rendered prompts; the server must not second-guess them.

    If the server applied its own chat template here, the bytes measured during evaluation would
    differ from the bytes the model was trained on, and the comparison would be meaningless.
    """
    client, engine = make_client()
    raw = "<|user|>already rendered<|/user|><|assistant|>"
    resp = client.post("/v1/completions", json={"prompt": raw, "max_tokens": 16})

    assert resp.status_code == 200
    assert engine.seen[-1] == raw, "server modified a pre-rendered prompt"
    assert SYSTEM_PROMPT not in engine.seen[-1]


def test_completions_handles_batches():
    client, engine = make_client()
    body = client.post("/v1/completions", json={"prompt": ["a", "b", "c"]}).json()
    assert len(body["choices"]) == 3
    assert [c["index"] for c in body["choices"]] == [0, 1, 2]
    assert body["usage"]["completion_tokens"] == 15
    assert engine.seen[-3:] == ["a", "b", "c"]
