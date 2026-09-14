"""Out-of-domain detection for the guardrail layer.

Why a second mechanism
----------------------
NeMo's dialog rails match an incoming message against handwritten canonical forms. Measured on a
dev set disjoint from the OOD probes (``reports/eval/rails_threshold.json``), that works for
requests which *look like* support queries but must be declined, and fails for requests that are
simply off-topic:

    at the zero-false-refusal threshold      off_topic 0.077   injection 0.132
                                             authority 0.343   cannot_know 0.474

The reason is structural rather than a bad threshold. The off-topic space is unbounded -- octopus
facts, haiku, quantum physics -- so no finite list of examples covers it by proximity. The
question worth asking is not "is this close to an off-topic example?" but "is this far from every
real support query?", and there are 23,453 real support queries to anchor that against.

    at 1% false-refusal                      off_topic 0.983   injection 0.900

The two mechanisms are kept because they are complementary, not redundant. "Just waive the fee" is
*semantically in-domain* -- it is about an order -- so distance cannot catch it, and the canonical
form must. Conversely no canonical form list catches arbitrary trivia. Each rail is used where it
measures well and neither is asked to do the other's job.

Cost
----
One local ONNX embedding (fastembed, CPU) and an 810-row dot product. No GPU, no second model, no
LLM call. The GPU stays entirely with the served model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ANCHORS = Path("guardrails/anchors.json")

VETO_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""A second, multilingual index that can veto the primary rail.

``bge-small-en-v1.5`` is English-only, and it showed: 14 of 19 non-English *in-domain* support
queries ("Quisiera saber cuándo llega mi pedido") scored as off-domain. A Spanish- or
Hindi-speaking customer never reached the model. That is a systematic failure of a whole user
class, not an edge case.

Swapping wholesale to the multilingual model fixes it but halves off-topic recall (0.750 -> 0.400).
Requiring *both* models to call a message off-domain keeps the English numbers exactly
(recall 0.600, off_topic 0.750, the same 3/1109 dev false refusals) and takes non-English false
refusals from 14/19 to 2/19 -- strictly better than either model alone, which is why there are two.

Cost is one extra 0.22 GB ONNX model on CPU. Still no GPU, still no LLM call.
"""

VETO_THRESHOLD = 0.62
"""Above this, the multilingual model considers the message in-domain and blocks the rail."""

DEFAULT_THRESHOLD = 0.63
"""Tuned, not chosen by feel -- see ``reports/eval/rails_threshold.json`` and the sweep recorded in
``reports/eval/rails_threshold.json``.

The scale is NeMo's, not cosine: ``BasicEmbeddingsIndex`` reproduces Annoy's angular score
``1 - sqrt(2 - 2*cos) / 2``, so this corresponds to cosine ~0.69. Reading it as a cosine would
badly misjudge how strict the rail is, which is how the first guessed value ended up far tighter
than intended.

"""

MIN_CONTENT_TOKENS = 6
"""Below this many words, the rail abstains and the message goes to the model.

Not a style preference -- a measured bug fix. The first version rated "Hi", "when", "not happy"
and "quick question" as off-domain and answered them with an off-topic redirect, dropping the
``asks_for_clarification`` probe from 0.750 to 0.350. Those messages are far from every anchor
because they carry no information, not because they are off-topic, and an embedding distance
cannot tell those two apart. The right response to an uninformative message is to ask what the
customer needs, which is what the model now does; the rail must not pre-empt it.

Swept alongside the threshold, with the synthetic ``underspecified`` queries added to the
negatives so the dev set actually contains the case that broke.
"""

CACHE_DIR = Path("artifacts/rails")


@dataclass
class DomainVerdict:
    off_domain: bool
    score: float
    nearest: str
    threshold: float
    abstained: bool = False
    """True when the message was too short to judge, so the rail deferred to the model."""

    veto_score: float | None = None
    vetoed: bool = False
    """True when the primary model called it off-domain and the multilingual model overruled."""


def content_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


class DomainRail:
    """Flags queries that are far from every known in-domain customer query.

    The index and the scoring arithmetic come from ``nemoguardrails`` itself rather than a local
    reimplementation, so the threshold tuned here means the same thing the runtime applies.
    """

    def __init__(
        self,
        anchors_path: Path | str = DEFAULT_ANCHORS,
        threshold: float = DEFAULT_THRESHOLD,
        min_content_tokens: int = MIN_CONTENT_TOKENS,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_engine: str = "FastEmbed",
        veto_model: str | None = VETO_MODEL,
        veto_threshold: float = VETO_THRESHOLD,
    ) -> None:
        self.min_content_tokens = min_content_tokens
        self.veto_model = veto_model
        self.veto_threshold = veto_threshold
        self._veto_index = None
        self.anchors_path = Path(anchors_path)
        payload = json.loads(self.anchors_path.read_text())
        self.anchors: list[str] = payload["anchors"]
        self.threshold = threshold
        self._model = embedding_model
        self._engine = embedding_engine
        self._index = None

    def _cache_path(self, model: str | None = None) -> Path:
        """Cache key covers the anchor texts and the model, so a change to either invalidates it.

        Embedding several thousand anchors takes tens of seconds on CPU. Paying that on every
        server restart is pure waste, and silently reusing a stale matrix after the anchors change
        would be worse -- hence the content hash rather than a timestamp.
        """
        model = model or self._model
        h = hashlib.sha256(("\u0000".join(self.anchors) + "|" + model).encode()).hexdigest()
        return CACHE_DIR / f"anchors-{h[:16]}.npy"

    async def _build_index(self, model: str):
        from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex
        from nemoguardrails.embeddings.index import IndexItem

        index = BasicEmbeddingsIndex(embedding_model=model, embedding_engine=self._engine)
        cache = self._cache_path(model)
        if cache.exists():
            index._items = [IndexItem(text=a, meta={}) for a in self.anchors]
            index.load(str(cache))
        else:
            await index.add_items([IndexItem(text=a, meta={}) for a in self.anchors])
            await index.build()
            cache.parent.mkdir(parents=True, exist_ok=True)
            index.save(str(cache))
        return index

    async def _ensure(self):
        if self._index is None:
            self._index = await self._build_index(self._model)
        if self.veto_model and self._veto_index is None:
            self._veto_index = await self._build_index(self.veto_model)
        return self._index

    @staticmethod
    async def _scores(index, texts: list[str]) -> np.ndarray:
        embs = np.asarray(await index._get_embeddings(texts), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cos = (embs / norms) @ index._index.T
        best = cos.max(axis=1)
        return 1.0 - np.sqrt(np.clip(2.0 - 2.0 * best, 0.0, None)) / 2.0

    async def score_many(self, texts: list[str]) -> np.ndarray:
        """Similarity of each text to its nearest in-domain anchor, on NeMo's angular scale."""
        index = await self._ensure()
        return await self._scores(index, texts)

    async def veto_many(self, texts: list[str]) -> np.ndarray:
        """The same, under the multilingual index."""
        await self._ensure()
        return await self._scores(self._veto_index, texts)

    async def check(self, text: str) -> DomainVerdict:
        if content_tokens(text) < self.min_content_tokens:
            return DomainVerdict(False, 0.0, "", self.threshold, abstained=True)
        index = await self._ensure()
        score = float((await self._scores(index, [text]))[0])

        embs = np.asarray(await index._get_embeddings([text]), dtype=np.float32)
        embs /= max(float(np.linalg.norm(embs)), 1e-12)
        nearest = self.anchors[int(np.argmax(index._index @ embs[0]))]

        off = score < self.threshold
        veto_score = None
        vetoed = False
        if off and self._veto_index is not None:
            veto_score = float((await self._scores(self._veto_index, [text]))[0])
            # The multilingual model recognises it as in-domain -- most often a support request
            # in another language -- so the English-only verdict is overruled.
            if veto_score >= self.veto_threshold:
                off, vetoed = False, True

        return DomainVerdict(
            off_domain=off,
            score=round(score, 4),
            nearest=nearest,
            threshold=self.threshold,
            veto_score=None if veto_score is None else round(veto_score, 4),
            vetoed=vetoed,
        )

    def check_sync(self, text: str) -> DomainVerdict:
        return asyncio.run(self.check(text))
