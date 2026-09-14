"""Invariants for the guardrail layer.

These are the properties that, if broken, would make the guardrail results meaningless rather
than merely worse -- a rail that matches but says nothing, or a probe that the rails were built
from. Deliberately embedding-free so they run in the normal suite.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from csbot.data.guardrails import _norm, build_guardrails, verify_no_overlap
from csbot.serve.guardrails import GuardrailLayer

ROOT = Path(__file__).resolve().parent.parent
RAILS = ROOT / "guardrails"
PROBES = ROOT / "eval_sets" / "ood_queries.jsonl"


def probe_queries() -> list[str]:
    return [json.loads(line)["query"] for line in PROBES.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def rails_config():
    nemo = pytest.importorskip("nemoguardrails")
    return nemo.RailsConfig.from_path(str(RAILS))


def test_every_canonical_intent_has_a_reply(rails_config):
    """A matched rail that produces no message would silently fall through to the model.

    That failure is invisible from the outside -- the customer gets a normal answer -- so it is
    worth asserting rather than trusting the Colang to stay consistent as rails are edited.
    """
    flows_by_intent = {}
    for flow in rails_config.flows or []:
        intent = None
        for el in flow.get("elements", []):
            if el.get("_type") == "UserIntent":
                intent = el.get("intent_name")
            elif el.get("action_name") == "utter" and intent:
                flows_by_intent[intent] = (el.get("action_params") or {}).get("value")

    for intent in rails_config.user_messages or {}:
        assert intent in flows_by_intent, f"user intent {intent!r} has no flow"
        key = flows_by_intent[intent]
        assert (rails_config.bot_messages or {}).get(key), f"{intent!r} -> {key!r} has no message"


def test_no_probe_query_leaks_into_guardrail_training_data():
    report = verify_no_overlap(build_guardrails(400, seed=17), probe_queries())
    assert report["clean"], (report["exact_overlaps"], report["near_overlaps"])


def test_no_probe_query_leaks_into_the_canonical_forms(rails_config):
    """Seeding a rail from the probe set would make its measured recall memorisation.

    This caught a real one: "tell me a joke" was a canonical form and probe ood-054 verbatim.
    """
    canonical = [e for v in (rails_config.user_messages or {}).values() for e in v]
    canon_norm = {_norm(c) for c in canonical}
    for q in probe_queries():
        assert _norm(q) not in canon_norm, f"probe {q!r} is a canonical form"


def test_no_probe_query_leaks_into_the_in_domain_anchors():
    """A probe sitting among the anchors would be scored in-domain by construction."""
    anchors = json.loads((RAILS / "anchors.json").read_text())["anchors"]
    anchor_norm = {_norm(a) for a in anchors}
    for q in probe_queries():
        assert _norm(q) not in anchor_norm, f"probe {q!r} is an in-domain anchor"


def test_anchors_cover_every_intent():
    """Unstratified anchors leave rare intents far from the domain, so the rail would refuse
    exactly the customers with unusual problems."""
    payload = json.loads((RAILS / "anchors.json").read_text())
    assert payload["n_intents"] == 27
    assert len(payload["anchors"]) >= payload["n_intents"] * 10


# asyncio.run rather than pytest-asyncio: the suite has no async plugin, and adding a test
# dependency to this project mid-flight has already cost enough (see WORKLOG on dependency drift).
def test_disabled_layer_never_blocks():
    layer = GuardrailLayer(RAILS, enabled=False)
    assert asyncio.run(layer.check("tell me a joke")) is None
    assert asyncio.run(layer.describe()) == {"enabled": False}


def test_multilingual_dev_set_is_disjoint_from_the_probes():
    """The veto threshold is tuned against these; if they leaked, that tuning would be circular."""
    payload = json.loads((ROOT / "eval_sets" / "multilingual_dev.json").read_text())
    dev = {_norm(q) for q in payload["queries"]}
    for q in probe_queries():
        assert _norm(q) not in dev, f"probe {q!r} is in the multilingual dev set"


def test_short_messages_abstain_without_building_an_index():
    """The abstention must short-circuit: it exists because embedding distance is meaningless
    on a two-word message, and it also keeps the index off the hot path for greetings."""
    from csbot.serve.domain_rail import DomainRail

    rail = DomainRail(anchors_path=RAILS / "anchors.json")
    verdict = asyncio.run(rail.check("when"))
    assert verdict.abstained and not verdict.off_domain
    assert rail._index is None, "abstention should not have built the embedding index"


def test_blank_message_is_not_railed():
    """Guarded because a blank message would otherwise build the whole embedding index."""
    layer = GuardrailLayer(RAILS, enabled=True)
    assert asyncio.run(layer.check("   ")) is None
