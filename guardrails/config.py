"""Registers the out-of-domain rail as a NeMo custom action.

NeMo auto-loads this file from the rails directory and calls ``init(app)``. Registering the check
as an *action* rather than running it outside NeMo keeps one orchestration layer: the rails config
remains the single place that decides what is blocked and what is said in reply, and the decision
shows up in NeMo's own explain/trace output alongside the dialog rails.

The action performs no LLM call -- one local ONNX embedding and a dot product against the frozen
anchor set. See ``csbot.serve.domain_rail`` for why this mechanism exists next to the dialog rails.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from csbot.serve.domain_rail import DomainRail  # noqa: E402

_RAIL: DomainRail | None = None


async def check_in_domain(context: dict | None = None) -> bool:
    """True when the message is far from every known in-domain support query.

    Built lazily so importing the config is cheap; the embedding index is constructed once on the
    first request and reused for the life of the process.
    """
    global _RAIL
    if _RAIL is None:
        _RAIL = DomainRail(anchors_path=Path(__file__).resolve().parent / "anchors.json")
    text = (context or {}).get("user_message") or ""
    if not text.strip():
        return False
    verdict = await _RAIL.check(text)
    return verdict.off_domain


def init(app) -> None:
    app.register_action(check_in_domain, "check_in_domain")
