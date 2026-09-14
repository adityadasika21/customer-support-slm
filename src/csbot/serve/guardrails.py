"""The guardrail layer: NeMo decides policy, the fine-tuned model does all the generating.

Why it is split this way
------------------------
The obvious integration is ``LLMRails.generate()``, letting NeMo drive. That was tried and
measured, not assumed. Pointing NeMo at a recording backend and sending one ordinary customer
query -- ``"I need to cancel order 884213 please"`` -- produced **three** LLM calls, each carrying
a prompt NeMo had built itself (1.3 KB and 3.4 KB), stuffed with generic scaffolding like
"As an AI assistant, I can help you with a wide range of tasks".

Two things break at once:

1. **Latency.** Three generations per request on a power-limited 8 GB card.
2. **Template drift.** The model was fine-tuned on exactly one prompt format, owned by
   ``csbot.serve.template`` and shared by training, evaluation and serving. A response produced
   under NeMo's format is not the model that was measured. Every format metric in the report --
   hygiene 0.995, the placeholder and entity numbers -- describes the trained template.

So NeMo keeps the parts it is genuinely good at and the parts that define policy:

* Colang parsing, canonical user forms and their example phrasings,
* the predefined bot messages returned when a rail fires,
* the embedding index and similarity threshold used to match them,
* the custom-action mechanism the out-of-domain rail registers through.

and generation never passes through it. A request either stops at a rail, with reviewable canned
text and no generation at all, or it reaches the model through the one canonical template.

This is the honest version of "NeMo Guardrails is integrated": it owns the policy definition and
the matching, it does not own the prompt.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("csbot.guardrails")

DEFAULT_CONFIG = Path("guardrails")


@dataclass
class RailDecision:
    """A request stopped by a rail."""

    rail: str
    """``out_of_domain`` for the input rail, otherwise the Colang user intent that matched."""

    message: str
    """Predefined Colang response. Never generated, so it is reviewable ahead of time."""

    score: float
    stage: str  # "input" | "dialog"


class GuardrailLayer:
    """Two complementary checks, both local embeddings, neither calling an LLM.

    ``input``  -- distance from the in-domain anchor set. Catches the open-ended off-topic space
                  (trivia, coding help, injections) that no finite example list can cover.
    ``dialog`` -- nearest canonical form from ``rails.co``. Catches requests that *are*
                  domain-shaped but must still be declined ("just waive the fee").

    See ``csbot.serve.domain_rail`` and ``reports/eval/rails_threshold.json`` for the sweep that
    produced this split, including why neither mechanism alone is sufficient.
    """

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG, enabled: bool = True) -> None:
        self.config_path = Path(config_path)
        self.enabled = enabled
        self._index: Any = None
        self._intent_to_message: dict[str, str] = {}
        self._threshold: float = 1.0
        self._domain: Any = None
        self._ready = False

    # -- construction ------------------------------------------------------------------

    async def _build(self) -> None:
        if self._ready or not self.enabled:
            return
        from nemoguardrails import RailsConfig
        from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex
        from nemoguardrails.embeddings.index import IndexItem

        from csbot.serve.domain_rail import DomainRail

        cfg = RailsConfig.from_path(str(self.config_path))
        dialog = getattr(cfg.rails, "dialog", None)
        user_messages = getattr(dialog, "user_messages", None) if dialog else None
        self._threshold = getattr(user_messages, "embeddings_only_similarity_threshold", 0.76)

        # Walk the parsed flows to learn which bot message answers which user intent, rather
        # than duplicating that mapping here. Editing rails.co stays sufficient to change policy.
        for flow in cfg.flows or []:
            intent = None
            for element in flow.get("elements", []):
                if element.get("_type") == "UserIntent":
                    intent = element.get("intent_name")
                elif element.get("action_name") == "utter" and intent:
                    key = (element.get("action_params") or {}).get("value")
                    msgs = (cfg.bot_messages or {}).get(key) or []
                    if msgs:
                        self._intent_to_message[intent] = msgs[0]

        emb = next(m for m in cfg.models if m.type == "embeddings")
        index = BasicEmbeddingsIndex(embedding_model=emb.model, embedding_engine=emb.engine)
        await index.add_items(
            [
                IndexItem(text=example, meta={"intent": intent})
                for intent, examples in (cfg.user_messages or {}).items()
                for example in examples
            ]
        )
        await index.build()
        self._index = index

        self._domain = DomainRail(anchors_path=self.config_path / "anchors.json")
        await self._domain._ensure()

        self._ready = True
        LOG.info(
            "guardrails ready: %d canonical forms, %d anchors, dialog threshold %.3f, "
            "domain threshold %.4f",
            len(index._items), len(self._domain.anchors), self._threshold,
            self._domain.threshold,
        )

    # -- the check ---------------------------------------------------------------------

    async def check(self, user_message: str) -> RailDecision | None:
        """Thread-offloaded: see ``_check``.

        The embedding lookups are CPU-bound ONNX calls. Run directly in the event loop they
        block every other in-flight request, which is what pinned served throughput at 40 tok/s
        against vLLM's 957.
        """
        return await asyncio.to_thread(self._check_sync, user_message)

    def _check_sync(self, user_message: str) -> RailDecision | None:
        return asyncio.run(self._check(user_message))

    async def _check(self, user_message: str) -> RailDecision | None:
        """Return the rail that stops this request, or ``None`` to let it reach the model.

        ``None`` is the common and the safe path. Wrongly refusing a real customer is a worse
        failure than occasionally answering something off-topic, so both thresholds were tuned
        against real validation queries with that asymmetry as the selection rule.
        """
        if not self.enabled or not (user_message or "").strip():
            return None
        await self._build()

        verdict = await self._domain.check(user_message)
        if verdict.off_domain:
            message = self._intent_to_message.get("ask off topic")
            if message:
                return RailDecision("out_of_domain", message, verdict.score, "input")

        results = await self._index.search(
            text=user_message, max_results=1, threshold=self._threshold
        )
        if results:
            intent = results[0].meta["intent"]
            message = self._intent_to_message.get(intent)
            if message:
                return RailDecision(intent, message, self._threshold, "dialog")

        return None

    async def describe(self) -> dict:
        """Configuration summary for ``/model-info``, so a reviewer can see what is enforced."""
        await self._build()
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "config": str(self.config_path),
            "engine": "nemoguardrails (policy only; generation is not routed through it)",
            "input_rail": {
                "mechanism": "distance from in-domain anchor set",
                "anchors": len(self._domain.anchors),
                "threshold": self._domain.threshold,
            },
            "dialog_rail": {
                "mechanism": "nearest Colang canonical form",
                "canonical_forms": len(self._index._items),
                "intents": sorted(self._intent_to_message),
                "threshold": self._threshold,
            },
            "llm_calls_per_check": 0,
        }
