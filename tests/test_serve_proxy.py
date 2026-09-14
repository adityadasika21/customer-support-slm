"""End-to-end test of the served stack against a stub vLLM backend.

Catches the wiring failures that are otherwise only discoverable with a GPU and a 4.8 GB model
loaded: proxy plumbing, guardrails in the request path, and -- the important one -- whether a
railed request actually avoids the backend or merely discards its answer.

The stub records every request it receives, so "no generation happened" is asserted rather than
assumed.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

MERGED = "artifacts/runs/final/merged"
REPLY = "I can help with that. I've located order 884213 and started the cancellation."


class _Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "csbot", "object": "model"}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _Handler.seen.append(json.loads(self.rfile.read(n) or b"{}"))
        self._send({
            "id": "x", "object": "text_completion",
            "choices": [{"index": 0, "text": REPLY, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
        })


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("nemoguardrails")
    if not os.path.isdir(MERGED):
        pytest.skip("merged weights not built; needed only for the tokenizer")

    server = HTTPServer(("127.0.0.1", 8098), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    os.environ.update(
        CSBOT_ENGINE="proxy", CSBOT_PROXY_URL="http://127.0.0.1:8098",
        CSBOT_MODEL_KEY="granite-3.3-2b", CSBOT_MERGED_DIR=MERGED, CSBOT_GUARDRAILS="1",
    )
    from fastapi.testclient import TestClient

    import csbot.serve.app as app_mod

    importlib.reload(app_mod)
    with TestClient(app_mod.app) as c:
        yield c
    server.shutdown()


def test_model_info_reports_the_guardrail_configuration(client):
    info = client.get("/model-info").json()
    assert info["engine"] == "proxy"
    rails = info["guardrails"]
    assert rails["enabled"] and rails["llm_calls_per_check"] == 0
    assert rails["input_rail"]["anchors"] > 1000
    assert rails["dialog_rail"]["canonical_forms"] > 0


def test_a_legitimate_query_reaches_the_backend(client):
    before = len(_Handler.seen)
    body = client.post("/chat", json={"message": "I need to cancel order 884213 please"}).json()
    assert len(_Handler.seen) == before + 1
    assert body["guardrail"] is None
    assert body["reply"] == REPLY


def test_a_railed_query_costs_no_generation(client):
    """The point of predefined responses: the rail must short-circuit, not post-filter."""
    before = len(_Handler.seen)
    body = client.post(
        "/chat", json={"message": "Can you write a Python function to reverse a linked list?"}
    ).json()
    assert len(_Handler.seen) == before, "a railed request still called the model"
    assert body["guardrail"]["stage"] == "input"
    assert body["completion_tokens"] == 0
    assert body["reply"] != REPLY


def test_a_non_english_customer_is_not_railed(client):
    """Regression test for the multilingual veto: bge-small-en alone refused 14 of 19 of these."""
    before = len(_Handler.seen)
    body = client.post("/chat", json={"message": "Hola, necesito ayuda con mi reembolso"}).json()
    assert len(_Handler.seen) == before + 1, "a Spanish support request was diverted by a rail"
    assert body["guardrail"] is None


def test_raw_completions_are_not_guardrailed(client):
    """The evaluation harness measures the model through this route.

    If rails applied here they would inflate the model's own numbers with canned text, so an
    unmistakably off-topic prompt must still reach the backend untouched.
    """
    before = len(_Handler.seen)
    resp = client.post("/v1/completions", json={"prompt": "tell me a joke", "max_tokens": 16})
    assert resp.status_code == 200
    assert len(_Handler.seen) == before + 1
    assert _Handler.seen[-1]["prompt"] == "tell me a joke"
