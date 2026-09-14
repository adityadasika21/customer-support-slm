"""The prompt template -- the single source of truth for training, evaluation and serving.

Train/serve template drift is the classic silent failure in a fine-tuning project: the model looks
excellent in the training notebook and quietly degrades behind the API because of one missing
newline or a different system prompt. Every component imports the format from here, and
``tests/test_template.py`` asserts that what the server sends byte-matches what the trainer fed
the model for the same example.

Data is emitted as ``{"prompt": ..., "completion": ...}`` pairs rather than a single concatenated
string. TRL consumes that shape directly and masks the loss to the completion, so the model is
never trained to generate the *customer's* turn -- only the assistant's reply.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

#: The system prompt used for training and serving. Kept short and behavioural: it describes the
#: role, not the 27 intents. Enumerating intents here would push the model toward classifying into
#: a closed set, which is exactly the template-memorisation failure we are trying to avoid.
SYSTEM_PROMPT = (
    "You are a customer support assistant. Read the customer's message, understand what they "
    "need, and reply accurately and helpfully in a professional, empathetic tone. Give concrete "
    "next steps when they apply, and never invent account details, order numbers, or policies "
    "you were not given."
)

#: Fraction of training examples rendered with no system prompt at all.
#:
#: This model will be run against held-out data, quite possibly with a differentwn
#: system prompt or none. A model trained with our system prompt on 100% of examples can become
#: dependent on those exact tokens. Dropping it on a minority of examples teaches the support
#: behaviour as a property of the weights rather than a response to one specific string, at
#: negligible cost. Applied at training time only -- evaluation and serving always use the
#: documented template.
SYSTEM_PROMPT_DROPOUT = 0.15


class ChatTokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this module depends on."""

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class TemplateOptions:
    """Per-model rendering quirks, sourced from ``csbot.models.registry.ModelSpec``."""

    supports_system_role: bool = True
    """Some chat templates reject a ``system`` turn; it gets folded into the user turn instead."""

    chat_template_kwargs: dict[str, Any] | None = None
    """Extra kwargs for ``apply_chat_template`` -- notably ``enable_thinking=False`` for the
    hybrid-reasoning models (Qwen3, SmolLM3). A support assistant must never emit a reasoning
    trace to a customer."""

    def kwargs(self) -> dict[str, Any]:
        return dict(self.chat_template_kwargs or {})


def build_messages(
    instruction: str,
    *,
    system: str | None = SYSTEM_PROMPT,
    options: TemplateOptions | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for a customer request.

    When the model has no ``system`` role, the system text is prepended to the user turn so the
    same instruction reaches the model either way.
    """
    options = options or TemplateOptions()
    instruction = instruction.strip()

    if not system:
        return [{"role": "user", "content": instruction}]

    if options.supports_system_role:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
    return [{"role": "user", "content": f"{system}\n\n{instruction}"}]


def render_prompt(
    tokenizer: ChatTokenizer,
    instruction: str,
    *,
    system: str | None = SYSTEM_PROMPT,
    options: TemplateOptions | None = None,
) -> str:
    """Render the full prompt string, ending at the point the assistant should start writing.

    This is what the server sends and what the trainer uses as the masked prefix.
    """
    options = options or TemplateOptions()
    messages = build_messages(instruction, system=system, options=options)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **options.kwargs(),
    )


def render_example(
    tokenizer: ChatTokenizer,
    instruction: str,
    response: str,
    *,
    options: TemplateOptions | None = None,
    eos_token: str = "",
    system: str | None = SYSTEM_PROMPT,
) -> dict[str, str]:
    """Render one training example as a prompt/completion pair.

    The EOS token is appended to the completion so the model learns where to *stop*. Without it a
    fine-tuned model will happily run on past the end of its answer and start a new one.
    """
    return {
        "prompt": render_prompt(tokenizer, instruction, system=system, options=options),
        "completion": response.strip() + eos_token,
    }


def render_training_example(
    tokenizer: ChatTokenizer,
    instruction: str,
    response: str,
    *,
    options: TemplateOptions | None = None,
    eos_token: str = "",
    rng: random.Random | None = None,
    dropout: float = SYSTEM_PROMPT_DROPOUT,
) -> dict[str, str]:
    """Render a training example, applying system-prompt dropout.

    Separate from :func:`render_example` so that the deterministic path used by evaluation and
    serving can never accidentally pick up the stochastic training behaviour.
    """
    rng = rng or random.Random()
    system = None if rng.random() < dropout else SYSTEM_PROMPT
    return render_example(
        tokenizer,
        instruction,
        response,
        options=options,
        eos_token=eos_token,
        system=system,
    )
