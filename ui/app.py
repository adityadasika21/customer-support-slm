"""Side-by-side: the base model and the fine-tuned model answering the same message.

Deliberately minimal -- one input, two outputs. Both arms get the same rendered prompt and the
same greedy decoding, and neither is guardrailed, so what you see is the difference the
fine-tune made and nothing else.

Unfilled `{{template slots}}` are highlighted. That is not decoration: a leaked slot is the
defect this project spent the most effort on, and it is easy to read straight past. Expect it
never to fire on a message you type yourself -- measured on the held-out set, neither the base
model nor the shipped fine-tune ever produces a slot from a prompt that did not contain one
(the raw-target model did, 706 times). To see the highlight work, paste a slot into the input.

No model code here. granite-3.3-2b declares no chat-template kwargs and supports a real system
role, so vLLM's chat endpoint renders exactly what `csbot.serve.template` renders. The system
prompt is fetched from the API rather than copied, so it cannot drift from training.

    bash scripts/serve_ui.sh
"""

from __future__ import annotations

import html
import os
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
import streamlit as st

# Defaults suit a local run; compose overrides them with service names (http://vllm:8001).
API = os.environ.get("CSBOT_API_URL", "http://127.0.0.1:8000")
VLLM = os.environ.get("CSBOT_VLLM_URL", "http://127.0.0.1:8001")
SLOT = re.compile(r"\{\{[^}]+\}\}")

st.set_page_config(page_title="Base vs Fine-tuned", page_icon="🎧", layout="wide")
st.markdown("""
<style>
  .slot{background:#fde68a;color:#7c2d12;padding:0 3px;border-radius:3px;font-weight:600;}
  .ans{font-size:0.95rem;line-height:1.6;white-space:pre-wrap;}
  .hdr{font-family:ui-monospace,monospace;font-size:.72rem;letter-spacing:.1em;
       text-transform:uppercase;padding:3px 10px;border-radius:5px;display:inline-block;
       margin-bottom:.6rem;}
  .b{background:#fbeee5;color:#a8501c;} .f{background:#e4f2ee;color:#16695a;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def setup() -> tuple[list[str], str]:
    """Served model names, and the system prompt the model was trained with."""
    models, system = [], ""
    try:
        models = [m["id"] for m in httpx.get(f"{VLLM}/v1/models", timeout=10).json()["data"]]
    except Exception:
        pass
    try:
        system = httpx.get(f"{API}/model-info", timeout=10).json()["system_prompt"]
    except Exception:
        pass
    return models, system


def ask(model: str, message: str, system: str) -> str:
    r = httpx.post(
        f"{VLLM}/v1/chat/completions",
        json={"model": model,
              "messages": ([{"role": "system", "content": system}] if system else [])
                          + [{"role": "user", "content": message}],
              "max_tokens": 400, "temperature": 0.0},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def show(text: str) -> str:
    return SLOT.sub(lambda m: f'<span class="slot">{m.group(0)}</span>', html.escape(text))


models, system = setup()

st.title("Base vs fine-tuned")
st.caption("Same message, same prompt template, greedy decoding. Neither side is guardrailed.")

if not models:
    st.error("vLLM is not running.")
    st.code("ADAPTER_DIR=artifacts/runs/final-polished/adapter bash scripts/serve_vllm.sh")
    if st.button("Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

if not system:
    # Deliberately fatal. Falling back to no system prompt is not a degraded comparison, it is a
    # different and misleading one: both arms then answer a bare user turn, which is not what the
    # model was trained or evaluated on, and it does not hurt the two sides equally.
    st.error("Cannot reach the API for the trained system prompt — refusing to compare without "
             "it, because an empty system prompt changes both answers and favours the fine-tune.")
    st.code("ADAPTER_DIR=artifacts/runs/final-polished/adapter bash scripts/serve_vllm.sh\n"
            "# or: docker compose -f docker/docker-compose.yml --profile ui up")
    # setup() is cached for 30s, so without this a browser refresh keeps showing the error for
    # half a minute after the API actually comes up -- which reads as "the UI is broken".
    if st.button("Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

if "base" not in models:
    st.warning(f"vLLM is serving {models} — there is no adapter loaded, so both columns would be "
               f"the same model. Start it with ADAPTER_DIR set (local) or MODEL=<base repo> "
               f"ADAPTER=<lora repo> (docker).")

base = "base" if "base" in models else models[0]
tuned = next((m for m in models if m != base), base)

message = st.text_area("Customer message",
                       value="can you cancel order 884213 for me", height=80)

if st.button("Compare", type="primary") and message.strip():
    with st.spinner("generating both"):
        # Both arms in parallel. Sequentially this is two full generations back to back --
        # about nine seconds of spinner with nothing on screen, which is a long time to watch.
        # vLLM batches the two requests into one forward pass, so in parallel it costs one.
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {name: pool.submit(ask, name, message, system)
                           for name in (base, tuned)}
                out = {name: f.result() for name, f in futures.items()}
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()

    left, right = st.columns(2, gap="medium")
    for col, name, css, label in ((left, base, "b", "Base model"),
                                  (right, tuned, "f", "Fine-tuned")):
        with col:
            st.markdown(f'<span class="hdr {css}">{label} · {name}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="ans">{show(out[name])}</div>', unsafe_allow_html=True)
