"""Tests for the prompt template -- the module every other component depends on.

The headline test here is ``test_served_prompt_matches_trained_prompt``: train/serve template
drift is the classic silent failure in a fine-tuning project, and it is invisible in metrics until
someone notices the deployed model is quietly worse than the notebook one.
"""

from __future__ import annotations

import random

import pytest

from csbot.serve.template import (
    SYSTEM_PROMPT,
    TemplateOptions,
    build_messages,
    render_example,
    render_prompt,
    render_training_example,
)


class FakeTokenizer:
    """Minimal stand-in with a deterministic, inspectable chat template.

    Using a fake rather than a real tokenizer keeps these tests fast and offline, and makes the
    assertions about structure rather than about one vendor's template quirks.
    """

    eos_token = "<|end|>"

    def apply_chat_template(self, conversation, *, tokenize=False, add_generation_prompt=False, **kw):
        parts = [f"<|{m['role']}|>{m['content']}<|/{m['role']}|>" for m in conversation]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        if kw.get("enable_thinking") is False:
            parts.append("<|nothink|>")
        return "".join(parts)


@pytest.fixture
def tok():
    return FakeTokenizer()


def test_system_role_used_when_supported(tok):
    messages = build_messages("where is my order", options=TemplateOptions(supports_system_role=True))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_system_folded_into_user_when_unsupported(tok):
    """Models whose chat template rejects a system turn must still receive the instruction."""
    messages = build_messages(
        "where is my order", options=TemplateOptions(supports_system_role=False)
    )
    assert [m["role"] for m in messages] == ["user"]
    assert SYSTEM_PROMPT in messages[0]["content"]
    assert "where is my order" in messages[0]["content"]


def test_no_system_prompt_when_none(tok):
    messages = build_messages("hello", system=None)
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == "hello"


def test_instruction_is_stripped(tok):
    messages = build_messages("   spaced out   ")
    assert messages[-1]["content"] == "spaced out"


def test_chat_template_kwargs_forwarded(tok):
    """Thinking mode must be disabled for reasoning models -- a customer must never see a trace."""
    options = TemplateOptions(chat_template_kwargs={"enable_thinking": False})
    assert "<|nothink|>" in render_prompt(tok, "hi", options=options)
    assert "<|nothink|>" not in render_prompt(tok, "hi", options=TemplateOptions())


def test_prompt_ends_at_generation_point(tok):
    """The prompt must stop exactly where the assistant should begin writing."""
    assert render_prompt(tok, "hi").endswith("<|assistant|>")


def test_render_example_appends_eos(tok):
    """Without EOS the model never learns where to stop and runs on past its answer."""
    example = render_example(tok, "hi", "Hello there.", eos_token=tok.eos_token)
    assert example["completion"] == "Hello there.<|end|>"
    assert example["prompt"].endswith("<|assistant|>")


def test_completion_excludes_prompt(tok):
    """Loss is computed on the completion only; the prompt must not leak into it."""
    example = render_example(tok, "where is my order", "It is on its way.")
    assert "where is my order" not in example["completion"]


def test_served_prompt_matches_trained_prompt(tok):
    """The bytes the server sends must equal the bytes the trainer masked as the prefix.

    This is the regression test for train/serve template drift. If someone changes the system
    prompt, the role handling, or the generation-prompt flag in one code path only, this fails.
    """
    instruction = "I need to cancel order 12345"
    options = TemplateOptions(chat_template_kwargs={"enable_thinking": False})

    trained = render_example(tok, instruction, "Sure thing.", options=options)["prompt"]
    served = render_prompt(tok, instruction, options=options)

    assert served == trained


def test_system_prompt_dropout_is_applied(tok):
    """Dropout must actually fire, so the model does not become dependent on our exact wording."""
    rng = random.Random(0)
    rendered = [
        render_training_example(tok, "hi", "hello", rng=rng, dropout=0.5)["prompt"]
        for _ in range(200)
    ]
    with_system = sum(1 for r in rendered if "<|system|>" in r)
    assert 0 < with_system < 200, "dropout produced a degenerate all-or-nothing split"


def test_training_dropout_disabled_matches_eval_path(tok):
    """With dropout off, the training renderer must agree exactly with the eval renderer."""
    rng = random.Random(0)
    trained = render_training_example(tok, "hi", "hello", rng=rng, dropout=0.0)
    evaluated = render_example(tok, "hi", "hello")
    assert trained == evaluated
