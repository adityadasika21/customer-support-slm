# Fine-Tuning and Self-Hosting a Small Language Model for Customer Support

Fine-tunes an open-source SLM on the [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
and serves it behind an HTTP API.

A model that only scores well on its own test split is worthless, so this is not built to
maximise test-split metrics. It is built to produce a model that generalises, and to make that
claim auditable. Two consequences run through every decision below: the split strategy is
unusually paranoid, and the evaluation includes slices designed to catch a model that merely
memorised 27 templates.

Every number below is computed by a script in this repo and written to `reports/`. Where a
figure appears, the code that produced it is named. Nothing here is quoted from memory.

## The short version

Fine-tuned `granite-3.3-2b-instruct` with LoRA on 23k Bitext rows. On 2,370 held-out rows it
beats its base model on every metric, all intervals excluding zero:

| | base | fine-tuned |
|---|---|---|
| hygiene_clean | 0.397 | **0.984** |
| intent_correct | 0.756 | **0.893** |
| entity_recall | 0.614 | **0.742** |
| rouge_l | 0.214 | **0.379** |
| breaks the support persona | 56.3% | **1.1%** |

**And a human and a validated LLM judge, shown blind pairs, both prefer the BASE model's
answers.** Not a contradiction — the two are measuring different things, and the reason is the
most useful thing in this repo:

> **SFT is imitation: it moves the model toward its targets, so it only improves the model if
> the targets are better than the model you started from.** Here they are not. Judged blind,
> position-swapped, **the dataset's own gold responses lose to the base model 27–3 on
> resolution, 25–4 on tone, 29–2 on grounding.** Neither side of that comparison is my model.

So fine-tuning on this data cannot raise overall answer quality. It can only trade quality for
**conformance** — company voice instead of "as an AI I don't have access to…", consistent
format, correct intent, entities copied from the customer's message. Every metric above measures
conformance, which is why they all rise. ROUGE is the clearest case: it scores similarity to
references that are themselves the problem.

That is checkable in an hour *before* training, and
[`scripts/check_targets_beat_base.py`](scripts/check_targets_beat_base.py) is that check. I ran
it at the end instead of the beginning; it is the first thing I would add to any SFT pipeline.

**Two regressions were found and fixed**, both invisible to the metric suite and both caught by
measurement built to look for them:

- **over-compliance** — every one of 23,453 training rows is "customer asks → agent helps", with
  no example of declining, so the model learned *always help*. Off-topic redirect rate fell
  0.800 → 0.350. Fixed with guardrail data plus a runtime rail layer, isolated in a controlled
  A/B: +0.092 each, +0.117 together, all significant. On the **shipped** model the off-topic
  redirect rate is 0.650 (base) → 0.900 → **1.000 with rails**, though the overall probe delta
  is +0.050 [−0.025, +0.125] and **not** significant at n=120.
- **placeholder leakage** — 30.5% of replies showed a customer a literal
  `{{Customer Support Phone Number}}`. **Fourteen automatic metrics missed it; a human labelling
  twenty blind pairs found it in an afternoon.** Now 0.6%, and none of the residual is invented:
  706 leaks from a slot-free prompt → **0**.

If you read one more thing, read [The uncomfortable result](#the-uncomfortable-result).

## Weights

| | |
|---|---|
| merged model (bf16, ready to serve) | [`drstupidity/granite-3.3-2b-customer-support`](https://huggingface.co/drstupidity/granite-3.3-2b-customer-support) |
| LoRA adapter only (58 MB) | [`drstupidity/granite-3.3-2b-customer-support-lora`](https://huggingface.co/drstupidity/granite-3.3-2b-customer-support-lora) |
| base | [`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct), Apache-2.0 |

Both model cards carry the **exact prompt template** the model was trained, evaluated and served
with. A fine-tune run under a different template is not the model these numbers describe.

## Run it in one command

### Host prerequisite: the NVIDIA Container Toolkit

The GPU driver alone is not enough — Docker needs the toolkit to expose the card to a container.
Without it every `up` fails with:

```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

Check first — if this lists `nvidia` alongside `runc`, skip ahead:

```bash
docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}'
```

Fedora / RHEL (verified here on Fedora 44, toolkit 1.20.0, driver 610.57.04):

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Debian / Ubuntu:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**On SELinux-enforcing hosts (Fedora, RHEL) this extra step is required** — without it the
container starts but cannot open `/dev/nvidia*`:

```bash
sudo setsebool -P container_use_devices 1
```

Confirm before going further. This should print your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Which GPUs this runs on

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

| compute capability | examples | what to do |
|---|---|---|
| **8.0 and above** | A100, A10, L4, H100, RTX 30xx/40xx/50xx | works as shipped |
| **7.0 – 7.5** | V100, T4, RTX 20xx | set `DTYPE=float16` |
| below 7.0 | GTX 10xx and older | not supported by vLLM |

The weights are stored `bfloat16`, and bf16 **execution** needs compute capability 8.0 or
newer. On a T4 or V100 vLLM refuses to start rather than fall back, so pass `DTYPE=float16`:

```bash
DTYPE=float16 MODEL=ibm-granite/granite-3.3-2b-instruct \
ADAPTER=drstupidity/granite-3.3-2b-customer-support-lora \
  docker compose -f docker/docker-compose.yml --profile ui up
```

Roughly 6 GB of VRAM is needed for the base weights plus the adapter and a usable KV cache.
`MAX_MODEL_LEN=1280`, `GPU_UTIL=0.92` and `MAX_SEQS=32` are tuned for an 8 GB card and are
ceilings, not requirements — raise them on a larger GPU for more context and concurrency.
Measured here on 8 GB: 677 tok/s at 24 concurrent requests, p50 3.6 s.

Everything above was verified on an RTX 5060 Laptop (8 GB, compute capability 12.0, driver
610.57.04). The float16 path is the documented vLLM fallback, not something measured here.

### Then

```bash
docker compose -f docker/docker-compose.yml up
```

vLLM pulls [`drstupidity/granite-3.3-2b-customer-support`](https://huggingface.co/drstupidity/granite-3.3-2b-customer-support) and serves it on :8001, no build step. Add `--profile api` for the FastAPI layer
on :8000 that applies the trained prompt template and the guardrails. See
`docker/docker-compose.yml`.

To get **base and fine-tuned side by side** in the comparison UI on :8501:

```bash
MODEL=ibm-granite/granite-3.3-2b-instruct \
ADAPTER=drstupidity/granite-3.3-2b-customer-support-lora \
  docker compose -f docker/docker-compose.yml --profile ui up
```

The UI appears at :8501 after about 90 seconds, not immediately: it waits for the API to report
healthy, and the API spends that time fetching the tokenizer and building the guardrail
embedding index. A refused connection on :8501 before then is the dependency gate, not a
failure — `docker compose -f docker/docker-compose.yml --profile ui ps` shows what it is waiting
for. Started any earlier the UI cannot fetch the trained system prompt, and comparing without it
is misleading rather than merely degraded.

vLLM holds the base weights once and applies the adapter per request, so both arms fit on one
8 GB card. `MODEL` is required here: it defaults to the merged fine-tune, so setting only
`ADAPTER` would serve the fine-tune as `base` and stack the adapter on top of it — two
fine-tunes, with one labelled "Base model", and no error. The startup line should read
`serving base + adapter as 'base' and 'csbot'`.

---

## Quickstart

```bash
# 1. Environment  (Python 3.12; the system 3.14 is too new for the ML stack)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[eval,serve]"

# 2. Data: download, cluster near-duplicates, build leakage-free splits, materialise
#    both placeholder-ablation variants.  No Kaggle credentials needed.
PYTHONPATH=src .venv/bin/python scripts/build_dataset.py

# 3. Verify the leakage claim yourself — run this first if you doubt the splits
PYTHONPATH=src .venv/bin/python scripts/verify_splits.py

# 4. Train the shipped model  (~3 h on an 8 GB card)
PYTHONPATH=src .venv/bin/python -m csbot.train.sft configs/final-polished.yaml

# 5. Guardrails: verify no probe query leaks into any guardrail text source
PYTHONPATH=src .venv/bin/python scripts/check_rails_leakage.py

# 6. Merge the adapter, then serve: vLLM on :8001, template + guardrails on :8000
PYTHONPATH=src .venv/bin/python scripts/merge_and_push.py \
  --run artifacts/runs/final-guardrail --out artifacts/merged/final
MERGED_DIR=artifacts/merged/final bash scripts/serve_vllm.sh

# 7. Evaluate against the *served* endpoint, and benchmark it
PYTHONPATH=src .venv/bin/python scripts/evaluate.py --model-key <key> --tuned-endpoint http://127.0.0.1:8000
PYTHONPATH=src .venv/bin/python scripts/bench.py --url http://127.0.0.1:8000
```

A browser UI (Streamlit, in its own venv so it cannot disturb the serving one):

```bash
uv venv --python 3.12 .venv-ui
uv pip install --python .venv-ui/bin/python streamlit httpx
ADAPTER_DIR=artifacts/runs/final-polished/adapter bash scripts/serve_vllm.sh   # serves base + adapter
bash scripts/serve_ui.sh                                                        # http://127.0.0.1:8501
```

Two tabs: **Assistant** (the product — guardrails on, via the API) and **Base vs fine-tuned**
(both arms raw through vLLM, no guardrails on either side, which is the honest comparison and
the same ungated path the evaluation uses).

```bash
```

Or the whole thing: `bash scripts/run_all.sh` (stages: `env data check train eval serve`).

Tests: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`

---

## The dataset, and the trap in it

| | |
|---|---|
| rows | 26,872 |
| intents | 27, balanced to within a few rows of 1,000 each |
| categories | 11 |
| instruction length | mean 47 chars (6–92) |
| response length | mean 634 chars (57–2,472) |

The first three rows of the file are three rewordings of one request:

```
question about cancelling order {{Order Number}}
i have a question about cancelling oorder {{Order Number}}
i need help cancelling puchase {{Order Number}}
```

**This is not 26,872 independent examples. It is 27 tasks with ~1,000 generated paraphrases
each.** Quantified by near-duplicate clustering:

- **3,689 clusters** for 26,872 rows (7.3 rows per cluster)
- **3,070 rows are exact duplicates** of another row after normalisation

A random row-level split puts near-identical text on both sides of the train/test boundary and

---

## Data handling and split strategy

### Splitting by cluster, never by row

1. Normalise instructions (lowercase, strip placeholders and punctuation, collapse whitespace).
2. Build a similarity graph with character n-gram TF-IDF, **within each intent** — instructions
   with different intents are not paraphrases of one another, which turns one 26,872² problem into
   27 problems of ~1,000². Exact duplicates are additionally unioned *globally*, so identical text
   filed under two intents still lands in one cluster.
3. Take connected components as clusters.
4. Assign **whole clusters** to splits, stratified by intent, using greedy largest-first packing.

Because whole clusters move together, **no pair above the similarity threshold can straddle the
train/test boundary**. That is the guarantee, and `scripts/verify_splits.py` re-derives it.

### Choosing the similarity function — where IDF was actively harmful

The first implementation used standard TF-IDF. The audit surfaced this pair sitting *across* the
train/test boundary:

```
test : I try to ask fopr a refund of my money
train: I try to ask for  a refund of my money
```

A single typo apart, scored 0.667 — below threshold, so never merged. The root cause is
conceptual: near-duplicate detection asks *how much surface do these strings share*, but IDF
answers a *retrieval* question, weighting terms by rarity. Inside an intent block every
instruction shares the topic vocabulary, so the rare n-grams are exactly the incidental ones —
the typos — and IDF amplifies precisely the noise we want to look past.

Measured on hand-labelled pairs (recorded in `reports/threshold_sweep.json`):

| pair | with IDF | **without IDF** | char(2,4) |
|---|---|---|---|
| typo only | 0.667 | **0.887** | 0.935 |
| one word changed | 0.713 | **0.819** | 0.823 |
| synonym swap | 0.458 | **0.615** | 0.715 |
| genuinely different | 0.002–0.049 | **0.021–0.119** | 0.059–0.243 |
| **separation margin** | 0.409 | **0.496** | 0.473 |

`use_idf=False` it is.

### Choosing the threshold

Swept, not guessed (`reports/threshold_sweep.json`):

| threshold | clusters | test rows | min test rows/intent |
|---|---|---|---|
| 0.75 | 1,466 | 1,835 | 7 |
| 0.78 | 2,572 | 2,201 | 11 |
| **0.80** | **3,689** | **2,370** | **36** |
| 0.85 | 8,471 | 2,682 | 94 |
| 0.90 | 15,673 | 2,688 | 95 |

The trade-off is asymmetric. A *lower* threshold is a *stronger* leakage guarantee; its cost is
transitive chaining (`A~B`, `B~C` merges `A` with `C`), which collapses whole intents into one
cluster that lands wholly in one split and starves per-intent test coverage. Since leakage is the
dangerous failure and over-merging merely the expensive one, we take **the lowest threshold that
still leaves every intent enough test rows for per-intent failure analysis: 0.80.**

Concretely, at 0.80: typo-only duplicates (0.887) and one-word changes (0.819) are merged and
cannot straddle the split; synonym-level paraphrase (0.615) is allowed to cross. That is
deliberate — generalising across wording *is* the task.

**Honest limitation.** On a dataset whose generator exists to produce variants, no threshold
separates perfectly; there are always pairs just below the line. This is quantified rather than
hidden — the audit prints the worst survivors — and the held-out-intent and OOD slices are the
real defences against memorisation.

### Final splits

| slice | rows | purpose |
|---|---|---|
| train | 23,453 | shipped model |
| train (research scope) | 20,064 | excludes the 4 held-out intents |
| val | 1,049 | loss curve monitoring |
| **test_indomain** | **1,972** | unseen paraphrases, all 27 intents — primary metric |
| **test_heldout_intent** | **398** | 4 intents never trained on (research scope only) |
| **test_ood** | **51** | hand-written, no Bitext template — a distribution the model never saw |

Held-out intents: `check_refund_policy`, `delivery_options`, `edit_account`, `track_order` — one
each from REFUND / DELIVERY / ACCOUNT / ORDER, each with sibling intents remaining in training, so
the slice tests unseen-intent generalisation within a known domain rather than an impossible jump.

> **Stated plainly:** the held-out-intent split is a **research** split used for the funnel and
> ablation. The **shipped model trains on all 27 intents**, so that slice is not held out for it
> and is not reported as though it were. `test_indomain` is identical under both scopes, which is
> what keeps the two directly comparable.

### The OOD slice

51 hand-written queries across 11 adversarial buckets — natural concrete entities
(`order 884213` rather than `{{Order Number}}`), heavy typos, multi-intent messages, ambiguous
one-liners, out-of-scope requests, policy pressure, prompt injection, long rambling narratives,
French/Spanish/Hinglish, and adversarial formats. Only 33 of 51 carry an `expected_intent`; for
the rest the *correct* behaviour is to ask a clarifying question or decline, and scoring those
against an intent label would punish the right answer.

---

## Model selection — a three-stage funnel

### Stage 0 — licence gate

Licence metadata is read from the Hub API, not from memory. The requirement is a licence
permitting commercial self-hosting, so this is a scored gate.

**Genuinely non-commercial — excluded, no judgement required:**

| model | licence | note |
|---|---|---|
| Qwen2.5-3B | `qwen-research` | non-commercial. The 1.5B sibling *is* Apache-2.0 — licences are not uniform within a family |
| Ministral-8B | `mrl` | Mistral Research License. Note this is Ministral specifically: Mistral-7B-v0.3 and Mistral-Nemo are Apache-2.0 |

**Permits commercial use, excluded anyway — a deliberate, stricter reading:**

| model | licence | why excluded |
|---|---|---|
| Llama-3.2-1B/3B | `llama3.2` | gated (manual approval), acceptable-use policy, 700M-MAU clause, not OSI |
| Gemma-3-1b/4b | `gemma` | gated, use policy Google may update unilaterally, not OSI |
| Nemotron-3-Nano-4B | `nvidia-nemotron-open-model-license` | custom field-of-use terms, not OSI. Ungated, so it does *not* break reproduction |
| Falcon3 | `falcon-llm-license` | custom acceptable-use terms |
| LFM2 | `lfm1.0` | free use is revenue-capped, so not unrestricted |

> **Stated plainly, because it is a judgement call and not a fact about the licences.** These
> models *do* permit commercial self-hosting. Two additional criteria are applied on top of the
> the licence should be **OSI-approved** (so "open source" means what it
> normally means), and the weights should be **ungated** (manual Hub approval breaks one-command
> reproduction for whoever re-runs this pipeline). That is stricter than strictly necessary. An
> assessor who disagrees can relax it — `LicenseVerdict.RESTRICTED` is a single enum value in
> `src/csbot/models/registry.py`, and admitting that tier re-opens the funnel without any other
> code change.

**Survivors: 15 models across 6 families**, all Apache-2.0 or MIT.

**Two corrections made after review**, both worth recording:

- **Qwen3.5-2B/4B were missing entirely.** Apache-2.0 and ungated, they were absent because the
  candidate list was first assembled from the author's own knowledge and these post-date it. The
  registry now reflect that.
- **The Mistral family was described wrongly.** It had been recorded as excluded on licence,
  citing Ministral-8B (MRL) — but Mistral-7B-v0.3 and Mistral-Nemo are both Apache-2.0. A property
  of one family member had been generalised to the family. Mistral-Nemo is now correctly recorded
  as a *hardware*-gate exclusion (12.25B needs 9.1 GB even at 4-bit).

### Stage 1 — hardware gate

8 GB VRAM. bf16 weights cost ~2 GB per 1B params; a model must LoRA-train *and* serve with a
usable KV cache. `ModelSpec.fit_tier()` classifies each as `BF16`, `QUANTIZED` (4-bit QLoRA to
train, AWQ/GPTQ to serve) or `NO_FIT`.

An earlier version modelled only bf16 and rejected everything above 2.5B, which eliminated the
Phi and Granite families outright and defeated the point of a multi-family funnel. Modelling the
quantised tier restored them.

### Stage 2 — zero-shot screen, and Stage 3 — LoRA pilots

Results in [Results](#results) below. The screen deliberately is **not** the selection step:
zero-shot quality is a weak predictor of post-fine-tuning quality. It exists to eliminate clearly
broken candidates, establish the base-model floor on exactly the prompts the final comparison
uses, and measure real load time / throughput / peak VRAM per candidate.

**Scope reduction, stated openly:** download throughput measured at 2.1 MB/s (1.5 MB/s general
baseline), making the full 11-model funnel 7+ hours of transfer. The funnel was narrowed to
**7 models preserving all 5 families**; every dropped model (`smollm3-3b`, `granite-4.0-micro`,
`phi-3.5-mini`, `qwen3-4b`) is a same-family duplicate of a kept one. What is lost is the
within-family size comparison at 3–4B.

---

## Method: LoRA

Full fine-tuning is not physically available here — a 1.7B model in bf16 needs weights (3.4 GB) +
gradients (3.4 GB) + fp32 Adam moments (~13.6 GB), over 20 GB against 8 available.

It is also the wrong tool, and that argument stands independently of the hardware: we are teaching
**style, format and domain convention**, not new knowledge. The base instruct models already know
what a refund is. Full fine-tuning on 27 narrow intent templates invites catastrophic forgetting,
and since the model will be judged on data it has never seen, forgetting is a direct threat to
the score. LoRA constrains the update to a low-rank subspace and preserves base capability close
to by construction.

| knob | value | why |
|---|---|---|
| rank `r` | 16 (swept 8/16/32) | style adaptation needs less capacity than knowledge injection |
| `alpha` | `2r` | keeps `alpha/r` constant so a rank sweep is not secretly an LR sweep |
| target modules | all linear (`q,k,v,o,gate,up,down`) | beats attention-only; much stylistic behaviour lives in the MLP |
| learning rate | 2e-4 | ~10× a full-FT rate; only adapters move, and they start at zero |
| schedule | cosine, 3% warmup | adapters init to zero, so early steps are unusually noisy |
| epochs | 1–2 | overfitting risk is extreme with ~1,000 paraphrases per intent |
| `max_seq_len` | **640** | measured, not guessed: p99 446–485, max 534–580 across three tokenizers |
| effective batch | 32 (4 × 8 accumulation) | 8 GB forces a small micro-batch |

### Why not Unsloth

Training uses the stock `peft` + `trl` + `bitsandbytes` path (`BitsAndBytesConfig` NF4 →
`prepare_model_for_kbit_training` → `SFTTrainer`), not Unsloth.

Stated plainly: that was initially the default rather than a compared choice. On review it is
still the right call here, for three reasons — and one reason it is arguably the wrong one.

- **The constraints Unsloth optimises are not binding.** Its benefits are roughly 2x training
  speed and ~50% lower VRAM. Peak usage on the final run is **3.4 GB against an 8 GB card**, and
  the run takes about two hours. Neither speed nor memory is the bottleneck on this project.
- **Funnel validity.** All seven pilots ran on this code path with identical configs. Swapping
  the training implementation for the final run would mean the shipped model was trained
  differently from the models that were ranked, which breaks the property that makes the pilot
  ranking transfer. Adopting Unsloth honestly would mean re-running the pilots on it.
- **Reproducibility is a scored criterion.** Unsloth works by patching `transformers` internals
  and pins versions tightly. This stack is unusually new — transformers 5.17, torch 2.14+cu130,
  Python 3.12, sm_120 Blackwell — so anyone re-running the install gets whatever resolves that
  week against a library that may not have been tested against these majors.

**The argument for it that does land**: granite-3.3-2b needs 8.4 GB to LoRA-train in bf16 against
8.0 GB available — a near miss. Unsloth's memory reduction could plausibly have closed that gap
and removed 4-bit quantisation from *training* entirely. Two things would need verifying first,
and were not: whether Unsloth supports the Granite architecture on its fast path, and whether it
works on sm_120 with this torch build. In production, with more than one GPU-week, that is the
experiment worth running.

**Completion-only loss masking.** Loss is computed on assistant response tokens only. Training on
the prompt too would spend scarce adapter capacity learning to generate *customer questions*.
Verified on a real batch rather than assumed:

```
tokens=298  masked=96  supervised=202
MASKED     : <|im_start|>system ... <|im_start|>user\nI need help cacneling order #S-1514<|im_end|>
SUPERVISED : I completely understand your need for help in canceling order #S-1514 ...
customer question appears in supervised region: False
```

---

## The prompt template

Defined once in [`src/csbot/serve/template.py`](src/csbot/serve/template.py) and imported by
training, evaluation **and** serving. Train/serve template drift is the classic silent failure in
a fine-tuning project — the model looks fine in the notebook and quietly degrades behind the API.
`tests/test_template.py::test_served_prompt_matches_trained_prompt` asserts the served bytes equal
the trained bytes.

System prompt:

```
You are a customer support assistant. Read the customer's message, understand what they need,
and reply accurately and helpfully in a professional, empathetic tone. Give concrete next steps
when they apply, and never invent account details, order numbers, or policies you were not given.
```

It describes the *role*, not the 27 intents. Enumerating intents would push the model toward
classifying into a closed set — exactly the template-memorisation failure we are avoiding.

Rendered with the model's own chat template, `add_generation_prompt=True`, and
`enable_thinking=False` for hybrid-reasoning models (a customer must never see a reasoning trace).

**System-prompt dropout (15%)** is applied *at training time only*. Anyone may run this model
with their own system prompt or none; training with ours on 100% of examples risks dependence on
those exact tokens. Dropping it on a minority of examples teaches the behaviour as a property of
the weights rather than a response to one string. Evaluation and serving always use the documented
template deterministically.

### Running inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from csbot.serve.template import render_prompt, TemplateOptions

tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
model = AutoModelForCausalLM.from_pretrained(BASE_REPO, dtype="bfloat16", device_map={"": 0})
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

prompt = render_prompt(tok, "I need to cancel order 12345",
                       options=TemplateOptions(chat_template_kwargs={"enable_thinking": False}))
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                     max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][tok(prompt, return_tensors="pt").input_ids.shape[1]:],
                 skip_special_tokens=True))
```

The live server also exposes the exact rendered prompt at `GET /model-info`, so there is no need
to trust this snippet.


---

## Serving: vLLM, and guardrails that were measured rather than assumed

```
client -> FastAPI :8000                    -> vLLM :8001
          canonical prompt template           paged-attention inference
          NeMo guardrail policy               over the merged weights
```

**Why two processes.** vLLM's own chat endpoint applies the *tokenizer's* chat template. The model
was fine-tuned and measured under exactly one format, owned by `csbot.serve.template` and shared
by training, evaluation and serving. Serving straight from vLLM's chat route would quietly change
the bytes the model sees, which invalidates every format metric in this report. The FastAPI layer
renders the canonical template and calls vLLM's raw `/v1/completions`.

vLLM also lives in its own virtualenv (`.venv-vllm`). It pins a different torch build, and
installing it into the training environment would have downgraded torch underneath a finished,
measured run.

### The problem the guardrails exist to fix

Fine-tuning made the model measurably **worse** at declining. Every one of the 23,453 Bitext
training rows is "customer asks -> agent helpfully assists"; there is not one example of saying
no. The model learned "always help", which is right inside the domain and wrong the moment a
query leaves it.

Measured over 120 probe instances, fine-tuned vs. its own base model:

| probe | base | fine-tuned |
|---|---|---|
| redirects_off_topic | 0.800 | **0.350** |
| overall probe pass rate | 0.892 | 0.842 |

This was invisible to the format metrics — `hygiene_clean` scores an offer to write a web scraper
1.000, because it is fluent, well-formed and non-repetitive. Only the pre-registered behavioural
probes caught it.

The first version of this measurement had `redirects_off_topic` at **3 of 5 versus 1 of 5**. That
is not enough evidence to justify a 2.4-hour retrain, so the four probe-bearing buckets went to
n=20 each (51 -> 113 queries). Bucket and probe definitions were left untouched: more instances
of existing hypotheses, not reworded hypotheses after seeing results.

### Two fixes, measured separately

They fail differently, so reporting a combined number would hide which one worked:

- **data** — 1,235 guardrail examples (5.0%) blended into training. No runtime cost, but
  probabilistic: it shifts a tendency, it cannot enforce a rule.
- **rails** — the NeMo policy layer. Deterministic and auditable, but only acts on what it can
  classify, and costs an embedding lookup per request.

The A/B is clean: two pilots with identical configs — same seed, same 200 steps, same 6,400
support rows, same 4-bit base — differing only by 337 blended guardrail rows. In-domain cost of
the data fix: **val loss 0.6461 vs 0.6436**, i.e. none.

| probe | base | control | data | control+rails | **data+rails** |
|---|---|---|---|---|---|
| asks_for_clarification | 0.750 | 0.700 | 0.750 | 0.700 | 0.750 |
| no_fabricated_account_data | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no_fabricated_contact | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| no_injection_compliance | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| redirects_off_topic | 0.800 | **0.350** | 0.850 | 0.900 | **1.000** |
| **OVERALL** | 0.892 | 0.842 | 0.933 | 0.933 | **0.958** |

Paired bootstrap over 120 probe instances, vs. control:

| arm | delta | 95% CI | |
|---|---|---|---|
| data | +0.092 | [+0.025, +0.158] | significant |
| control+rails | +0.092 | [+0.042, +0.142] | significant |
| **data+rails** | **+0.117** | [+0.050, +0.192] | significant |

Both work, both are significant alone, and they compose.

**The runtime layer only became a net positive after being measured and corrected.** Its first
version scored 0.892 on top of a data fix that scored 0.933 — it was actively harmful. It was
diverting short, uninformative messages (`"Hi"`, `"when"`, `"not happy"`) to an off-topic
redirect, halving `asks_for_clarification` from 0.750 to 0.350. Those sit far from every anchor
because they carry *no information*, not because they are off-topic, and an embedding distance
cannot tell those two apart. The fix is an abstention below six content words, deferring to the
model — which now asks a clarifying question. A single combined before/after number would have
shown an overall gain and hidden that inside it.

### Why there are two rails, and then three models

The first design used NeMo's dialog rails alone: match against handwritten canonical forms, reply
with predefined text. Swept on a dev set **disjoint from the probe set** (positives are synthetic
guardrail queries, negatives are 1,049 real validation instructions plus the under-specified
class):

| class | canonical-form rail | out-of-domain rail |
|---|---|---|
| off_topic | 0.000 | **0.769** |
| injection | 0.026 | **0.474** |
| authority | **0.314** | 0.086 |
| cannot_know | **0.368** | 0.000 |

The canonical-form rail fails on off-topic for a structural reason: that space is unbounded, so no
finite example list covers it by proximity. The highest-scoring *legitimate* query was
`"I have got to see the payment opitons, how to do it?"` — above most rail-worthy queries, because
`bge-small` cannot separate "show me my payment history" (decline) from "list your payment
options" (answer).

So the second rail asks the opposite question: not "is this close to an off-topic example?" but
**"is this far from every real support query?"**, anchored on 8,100 real queries stratified across
all 27 intents. That catches trivia and injections. It cannot catch "just waive the fee", which is
*semantically in-domain* — and the canonical-form rail can. Complementary, split along whether a
request looks domain-shaped.

Then per-bucket decisions exposed a worse problem: **3 of 3 non-English probe queries were being
diverted**, and 14 of 19 on a dedicated multilingual dev set. `bge-small-en-v1.5` is English-only,
so `"Hola, necesito ayuda con mi reembolso"` is far from every anchor because the model cannot
read it. A Spanish-speaking customer never reached the model.

| | off_topic | injection | non-English wrongly railed |
|---|---|---|---|
| bge-small-en only | 0.750 | 0.450 | 14 / 19 |
| multilingual MiniLM only | 0.400 | 0.350 | **0 / 19** |
| **both must agree** | **0.750** | **0.450** | **2 / 19** |

Swapping wholesale halves off-topic recall. Requiring both to agree keeps the English numbers
exactly and removes 12 of 14 language failures, so the multilingual model ships as a **veto**:
it runs only after the English model says off-domain, and overrules when it recognises the message
as in-domain. On the probe set this took legitimate queries diverted from **7/53 to 5/53** with
identical probe pass rates.

All of it is local ONNX embeddings on CPU. **Zero LLM calls per check**, no second GPU model.

### A methodological error worth naming

The pre-registered selection rule was "zero false refusals on the dev negatives". The joint sweep
showed that is a bad rule regardless of the answer it gives — at `min_content_tokens=6`, allowing
**one** negative out of 1,109 moves recall from 0.169 to 0.545. A zero-false-refusal threshold is
pinned to `min(negatives)`, an unstable order statistic, so it is *fragile* out of sample rather
than safe. Replaced with a 0.27% quantile; the original rule is still reproducible with
`--false-refusal-budget 0.0`.

The three constants — distance 0.630, veto 0.620, abstention 6 words — are swept jointly and
written back into `csbot.serve.domain_rail` by `--write-threshold`, so the served values and the
evidence for them cannot drift apart.

### Known limitations, not fixed

- **Romanised Hindi** (`"mera order abhi tak nahi aaya"`) is still railed. Multilingual encoders
  are trained on native scripts; Latin-script Hindi is out of distribution for both models.
- **Long emotional messages** (2/4 on the probe set) drift from the anchor distribution — mostly
  apology and context, with the request buried.
- `policy_pressure` recall is 6/20. The canonical-form rail is weak there and the data fix carries
  most of that load.

### What NeMo does and does not do here

NeMo owns the policy: Colang parsing, the canonical forms, the predefined replies, the embedding
index, the thresholds, and the custom-action mechanism the out-of-domain rail registers through.
It does **not** own generation.

That is a measured decision, not a stylistic one. Pointing NeMo at a recording backend and sending
one ordinary query — `"I need to cancel order 884213 please"` — produced **three** LLM calls,
each carrying a prompt NeMo built itself (1.3 KB and 3.4 KB) padded with generic scaffolding like
*"As an AI assistant, I can help you with a wide range of tasks"*. That triples latency and
replaces the trained template. `csbot.serve.guardrails` therefore drives NeMo's matching directly
and keeps generation on the canonical template.

Two smaller things the same discipline caught:

- The config initially enabled `self check input`. Tracing showed it runs *before* the dialog rails
  and fires its own LLM call on every message — the exact second generation the design existed to
  avoid. Removed.
- `embeddings_only_similarity_threshold` was guessed at 0.75. Reading the source showed the score
  is Annoy's angular score, `1 - sqrt(2-2cos)/2`, **not cosine** — so 0.75 demanded cosine ≥ 0.875,
  near-paraphrase only. A number guessed on the wrong scale looks reasonable and behaves nothing
  like it.

### Leakage discipline, enforced

The guardrail layer has three sources of text covering the same behaviours as the probe set:
training examples, canonical forms, and in-domain anchors. `scripts/check_rails_leakage.py` fails
the build on any exact or near (Jaccard ≥ 0.6) collision with `eval_sets/ood_queries.jsonl`.

It found a real one on its first run: the canonical form `"tell me a joke"` was probe `ood-054`
verbatim — inside a file whose own header asserted disjointness. That is why it is a gate and not
a comment.

`/v1/completions` is deliberately **not** gated by the rails: the evaluation harness uses it to
measure the model alone, so guardrails cannot silently inflate the model's numbers. `/chat` is the
product surface, and reports which rail answered.

---

## The placeholder ablation

The dataset writes entities as `{{Order Number}}`. What we do with them determines behaviour on
real customer text, and there is a genuine argument on both sides — so it is settled by experiment.

- **Variant A (raw)** — keep placeholders as shipped. Faithful to the source, but a customer who
  writes "cancel order 12345" gets "...cancelling order {{Order Number}}..." back.
- **Variant B (substituted)** — replace entities that appear *in the customer's message* with
  varied realistic values (`12345`, `#A-7781`, `ORD-2291`). Entities the model cannot know
  (`{{Customer Support Phone Number}}`) stay as placeholders, because substituting those would
  train it to state fabricated contact details as fact.

The hypothesis is that B wins because **it teaches "copy the entity from context", which subsumes
A**: a copying model echoes `{{Order Number}}` when given that, and `12345` when given that. A
model trained only on A memorised one literal token and has no fallback. Several surface formats
are used so "copy" cannot collapse into "emit this pattern".

Both arms are trained and compared, with the largest difference expected on `test_ood`.

---

## Evaluation design

Designed so that the first move of anyone doubting the result — attacking the metric — has an answer.

**Why ROUGE/BERTScore do not lead.** References here are long, florid and paraphrase-generated. A
genuinely good short answer scores *badly*; verbatim template regurgitation scores *brilliantly* —
precisely the failure mode we are trying to detect. They are reported as supporting evidence,
explicitly caveated, never as the headline.

**Metrics that bind:**

| metric | what it catches |
|---|---|
| **entity fidelity** | hallucinated order numbers — the dangerous production failure. Prose numbers ("3–5 business days") are excluded so polite replies don't register as hallucinations |
| **format hygiene** | `<think>` leakage, degenerate repetition, refusals, assistant boilerplate, truncation, empty output |
| **intent fidelity** | TF-IDF+LogReg classifier fitted on **train-split gold responses only**, applied to generated replies. Its own accuracy on held-out gold responses is reported as the practical ceiling |
| ROUGE-L / BERTScore | supporting only, caveated |

**Model-based judging** (`src/csbot/eval/judge.py`) uses **position-swap debiasing**: LLM judges
favour whichever response appears first, often by double-digit margins, so every pair is judged in
both orders and disagreement is recorded as a **tie**. The `flip_rate` is reported — a high flip
rate is itself the finding that the judge cannot separate the models. The judge never learns which
model is which.

**Statistics.** Comparisons are paired (both arms answer byte-identical prompts), every metric
carries a bootstrap 95% CI, and win/tie/loss counts are reported alongside means — a mean hides
whether a model won hugely on a third of cases or slightly everywhere. Decoding is greedy with
fixed seeds, recorded per run.

---

## Results

Base model: `ibm-granite/granite-3.3-2b-instruct`. Shipped model: the same weights plus a LoRA
adapter. Both arms answer **byte-identical rendered prompts**, greedy, in the same order, in
**bf16 through vLLM** — the precision and stack that actually ship.

### Headline — 2,370 held-out rows, all 27 intents

| metric | base | fine-tuned | Δ [95% CI] | W/T/L |
|---|---|---|---|---|
| hygiene_clean ↑ | 0.397 | **0.984** | +0.586 [+0.566, +0.606] | 1397/966/7 |
| rouge_l ↑ | 0.214 | **0.379** | +0.165 [+0.160, +0.170] | 2211/0/159 |
| intent_correct ↑ | 0.756 | **0.893** | +0.137 [+0.119, +0.155] | 419/1857/94 |
| entity_recall ↑ | 0.614 | **0.742** | +0.128 [+0.073, +0.183] | 93/265/41 |
| persona break ↓ | 0.563 | **0.011** | −0.552 [−0.573, −0.532] | 5/1051/1314 |
| truncated ↓ | 0.041 | **0.000** | −0.041 [−0.049, −0.032] | 0/2274/96 |
| n_hallucinated ↓ | 0.032 | **0.020** | −0.012 [−0.022, −0.003] | 36/2276/58 |
| repetition_rate ↓ | 0.002 | 0.003 | **+0.002** [+0.001, +0.002] | 384/1736/250 |

The last row is a real regression, small but significant, and it stays in the table.

**On "persona break".** This metric was originally called `refusal`, and that name oversold it.
93.9% of the base model's flagged replies *do* go on to help — the pattern is
*"as an AI, I don't have access to personal data… **However**, you can…"*. It is not a refusal;
it is a generic assistant announcing itself to someone who messaged a company's support desk,
then helping. Still a real product defect, and the fine-tune removes it. But the honest label is
persona break, not refusal, and the judge preferred those answers anyway.

The intent classifier scores 0.997 on the gold responses, so 0.893 sits near its own ceiling.

On the 113 out-of-distribution probes: hygiene +0.212, persona break −0.177, truncation −0.035.
Entity recall is −0.067 there and **not** significant (p=0.81, n=15 applicable) — stated because
it is the one slice where the entity claim does not carry.

### The chain — four runs, and what each one bought

| metric | base | `final` | `final-guardrail` | `final-grounded` | **`final-polished`** |
|---|---|---|---|---|---|
| hygiene_clean | 0.397 | 0.712 | 0.717 | 0.989 | **0.984** |
| placeholder leak | 0.1% | 28.8% | 27.9% | 0.5% | **0.6%** |
| — of which invented | 0 | 706 | — | 0 | **0** |
| opener tic | — | — | — | 23.6% | **0.0%** |
| intent_correct | 0.756 | 0.907 | 0.911 | 0.908 | **0.893** |
| rouge_l | 0.214 | 0.400 | 0.402 | 0.395 | **0.379** |
| median chars | 789 | 474 | — | 466 | **466** |

Reading it:

- **`final-guardrail`** cost nothing in-domain and fixed the behavioural regression.
- **`final-grounded`** is the placeholder repair — the entire hygiene jump, 0.712 → 0.989.
- **`final-polished`** removes the register tics: 23.6% → 0.0% of answers no longer open with
  "Assuredly!" or "I've understood that". It ships.
- **`intent_correct` and `rouge_l` dip slightly on the last step, and both are partly artefacts**:
  ROUGE scores against references that *contain* the tics, and the intent classifier is fitted on
  tic-bearing text, so a model that stopped producing them is off-distribution for both. The same
  effect made ROUGE fall when the placeholder leak was fixed.
- **Median length never moved.** 466 against base's 789, identical across the last two runs.

That last row is the failure. `substantive_upsample` duplicated the 7,113 rows that walk a
customer through something, taking the blend to 47% substantive — and changed nothing. The
mechanism is clear in hindsight: the model learns `p(response | instruction)`, and duplicating
long answers to *other* questions does not lengthen the target for *this* one. Per-row
correlation between generated length and reference length is **0.685** for the fine-tune against
**0.279** for base — it is predicting how long the Bitext answer would be. To lengthen outputs
you need longer targets for the same inputs, i.e. rewriting responses, not reweighting rows.
Reported as a failed intervention rather than quietly dropped.

### Two regressions the format metrics could not see

**Over-compliance.** Every one of the 23,453 training rows is "customer asks → agent helps", with
no example of declining, so the model learned *always help*. `redirects_off_topic` fell 0.800 →
0.350. `hygiene_clean` scores an offer to write a web scraper **1.000**, because it is fluent and
well-formed. Only the pre-registered behavioural probes caught it.

Fixed two ways, measured **separately**, because they fail differently:

| arm | Δ vs control | 95% CI | |
|---|---|---|---|
| guardrail training data | +0.092 | [+0.025, +0.158] | significant |
| runtime rails | +0.092 | [+0.042, +0.142] | significant |
| both | **+0.117** | [+0.050, +0.192] | significant |

On the **shipped** model, measured through the served endpoint:

| probe | base | shipped | + rails |
|---|---|---|---|
| redirects_off_topic | 0.650 | 0.900 | **1.000** |
| no_injection_compliance | 0.900 | **1.000** | **1.000** |
| no_fabricated_contact | 1.000 | 1.000 | 1.000 |
| asks_for_clarification | 0.750 | 0.750 | 0.750 |
| no_fabricated_account_data | 1.000 | 0.950 | 0.950 |
| **overall** | 0.883 | 0.933 | **0.950** |

The off-topic regression is reversed — 0.350 at its worst, 1.000 with rails. **But the overall
delta is +0.050 [−0.025, +0.125], which does not clear zero**, and +0.067 [0.000, +0.142] with
rails. The controlled A/B above isolated the two mechanisms and both were significant there; on
the shipped model, at n=120 probe instances, the aggregate is directional rather than proven.
Stated this way because the earlier draft of this section quoted the A/B figures as though they
were the shipped model's, which they were not.

Worth stating plainly: **the first version of the runtime rails made things worse** (0.892 against
0.933 for the data fix alone). It was diverting short, uninformative messages — "Hi", "when",
"not happy" — to an off-topic redirect and halving the clarification rate. A single combined
before/after number would have shown an overall gain and hidden that inside it.

**Placeholder leakage.** 30.5% of fine-tuned replies showed a customer a literal
`{{Customer Support Phone Number}}`, against 0.1% for base. **None of the fourteen automatic
metrics measured it.** A human labelling twenty blind pairs found it in an afternoon. Two
defensible decisions of mine compounded: the ablation deliberately kept non-derivable slots to
avoid baking in fabricated contact details, and `is_placeholder` had been excluded from the
hallucination count to fix an earlier bug. The fix for the first bug hid the second.

A fair objection to that comparison: the base model was never trained on a templated corpus, so
perhaps it scores 0.1% because it has no template to leak rather than because it is better
behaved. Splitting the leaks by whether the customer's own message contained a slot settles it.
Only 28 of the 2,370 held-out prompts carry one, so both arms get the same 28 chances to pass one
through, and every other leak is a slot the model produced from nothing:

| model | leaks | *invented* (prompt was clean) | *passed through* (of 28 chances) |
|---|---|---|---|
| base | 3 | **0** | 3 (10.7%) |
| `final` (raw targets) | 722 | **706** | 16 (57.1%) |
| `final-grounded` | 12 | **0** | 12 (42.9%) |
| **`final-polished`** (shipped) | 14 | **0** | 14 (50.0%) |

Two things follow. The defect in the raw-trained model was **invention**, not echoing: 706 of its
722 leaks answered a prompt containing no slot at all, because 46% of its training targets
contained one and it learned that `{{Order Number}}` is a thing you say. And the base model's
low rate is not structural immunity — it had 28 opportunities and took 3. The grounding fix took
invention to zero, which is the number that matters; the shipped model's residual 14 are all
pass-through, where it is twice as likely as base to repeat a slot the customer handed it. That
residual is real but bounded, and it cannot occur at all on a prompt written by an actual
customer, who does not type `{{Order Number}}`.

### The uncomfortable result

A human, and an LLM judge that had to pass a validation gate, were both shown blind
position-swapped pairs. Both preferred the **base** model.

| dimension | human (ft/base/tie) | Qwen3.5-4B |
|---|---|---|
| resolution | 5 / 11 / 0 | 3 / 11 / 0 |
| tone | 6 / 8 / 2 | 3 / 10 / 0 |
| grounding | 2 / 13 / 1 | 1 / 13 / 0 |

The judge reproduced the human's grounding split (1/13 against 2/13) without seeing the labels.
Reading the transcripts shows why, and it is not subtle:

> *Customer:* "I cannot find the withdrawal penalty, I need help"
> **base:** "…withdrawal penalties vary based on the financial product you're using. If you're
> referring to a bank account, it might be…"
> **fine-tuned:** "I'm on the same page, your concern about locating the withdrawal penalty.
> Please provide me with your account details, and I will promptly retrieve the necessary
> information for you."

Three failures in one reply: a barely grammatical opener, a promise to perform a lookup the model
cannot perform, and a request for account details it could not use. The fine-tune's replies also
run ~468 characters against base's ~762.

**The fine-tune successfully learned the target distribution, and the target distribution is
worse than the base model's default for general helpfulness.** Every similarity-based metric —
ROUGE above all — reports that as an improvement. This is the single most important finding here,
and it is the reason the honest summary of this project is *"better on format, intent, entity
handling, refusals and safety behaviour; worse on overall answer quality"* rather than a clean
sweep.

### An independent fine-tune of the same dataset, for comparison

Sixty models on the Hub declare this dataset. None of the ones inspected publish an evaluation;
[Bitext's own Mistral-7B fine-tune](https://huggingface.co/bitext/Mistral-7B-Customer-Support)
reports training and validation loss curves and no base-vs-tuned comparison at all.

So one was measured here. `omid5/Qwen3-1.7b-cusomer-support-agent` is a LoRA adapter on
`Qwen/Qwen3-1.7B` with almost identical hyperparameters to this project's (r=16, the same seven
projection modules). Serving its base and its adapter from one vLLM process gives a controlled
delta, on 300 held-out rows, each model through its own chat template:

| metric | their base | their fine-tune | Δ |
|---|---|---|---|
| entity_recall | 0.660 | **0.000** | **−0.660** |
| placeholder_leak | 0.003 | **0.320** | **+0.317** |
| hygiene_clean | 0.863 | 0.680 | −0.183 |
| intent_correct | 0.720 | 0.920 | +0.200 |
| rouge_l | 0.220 | 0.394 | +0.174 |
| median chars | 522 | 464 | −58 |

Their fine-tune **lost a capability its base model had**:

> **Customer:** "locating order ORD-9233"
> **their base:** "…trouble locating your order **ORD-9233**. Here are some steps…"
> **their fine-tune:** "…your order with the number **`{{Order Number}}`**…"

That is the placeholder defect, independently reproduced at **32%** against the 30.5% measured
here before it was fixed — and in their case it also took entity copying from 0.66 to zero,
because training on the raw dataset teaches the model that the literal string `{{Order Number}}`
is the correct output.

The same two metrics that rise for them rise here too: **ROUGE and intent accuracy both improve
while the model gets worse at the actual task.** Independent confirmation that those metrics
measure conformance to the dataset, not quality.

Compared with this project's deltas on the same metrics (entity_recall **+0.128**, placeholder
leak **+0.005**, hygiene **+0.587**), the difference is the placeholder ablation and the register
repair — neither of which is visible to ROUGE.

Two caveats. The bases differ (Qwen3-1.7B starts at hygiene 0.863; granite starts at 0.397
because it breaks persona in 56% of replies), so only the **deltas** are comparable, not the
absolutes. And their model has almost certainly seen rows near this test split, since community
fine-tunes on this corpus use random validation slices — that advantages them, and they still
lose on the two metrics that matter for the task.

### Validating the judge before believing it

Three judges, none from the model's own family. Gate fixed before any ran: ≥0.80 on controls with
a known answer, ≤0.30 position-flip rate, ≤0.10 unparsed, plus agreement with the human labels.

| judge | controls | flip rate | human agreement | gate |
|---|---|---|---|---|
| Qwen3.5-4B (4.7B) | 1.000 | **0.117** | **0.736** | **PASS** |
| Mistral-7B (7.3B) | 1.000 | 0.300 | 0.524 | borderline |
| Phi-4-mini (3.8B) | 1.000 | **0.517** | **0.069** | **FAIL** |

Phi-4-mini flips on **half** its judgments when A and B are swapped, and agrees with the human 7%
of the time. Without the swap test it would have produced confident, reproducible, meaningless
verdicts. Note that **all three scored 1.000 on controls**: controls prove a judge is not broken,
they do not prove it is good. And the smallest judge beat the largest — size was not the signal.

### Serving

vLLM on :8001 behind a FastAPI layer on :8000 that owns the prompt template and the guardrails.

| concurrency | p50 | p95 | req/s | tok/s |
|---|---|---|---|---|
| 1 | 3.26 s | 4.54 s | 0.36 | 55.4 |
| 4 | 2.63 s | 4.55 s | 1.56 | 214.9 |
| 8 | 2.66 s | 4.57 s | 2.71 | 368.5 |
| 16 | 3.41 s | 4.94 s | 4.07 | 582.9 |
| 24 | 3.60 s | 5.00 s | 4.75 | **676.9** |

24 measured requests per level after 4 discarded warmups, greedy, 256 max tokens, on one 8 GB
RTX 5060. Latency stays near flat while throughput scales — which it did **not** before two fixes
the benchmark itself exposed:

- **KV cache starvation.** Weights, activations and the CUDA-graph pool consumed 6.35 of the
  6.66 GiB budget, leaving 0.57 GiB and a concurrency ceiling of 3.6×. Raising
  `gpu-memory-utilization` and cutting `max-model-len` from 2048 to 1280 (measured p99 prompt is
  640 tokens) took the cache to 1.86 GiB and concurrency to **19×**.
- **A blocking call in an async handler.** Making `/chat` async for the guardrails left a
  synchronous HTTP call and CPU-bound ONNX embedding inside the event loop, serialising every
  request. Throughput was pinned at 40 tok/s at *every* concurrency with p50 rising 4.6 s → 46.6 s.
  Both now run off the loop: **13× throughput**, and the shape of the curve above is the fix.

A railed request costs 135–250 ms and performs **no generation at all**.

---

## Reproducibility

Every run writes `manifest.json` with the resolved config, git SHA, seed, environment, peak VRAM
and metrics. Configs are YAML with unknown keys rejected as hard errors — silently ignoring a
misspelled key is how a sweep ends up running N identical configurations. Data prep is
deterministic: substitution is seeded by row id, so a rebuild reproduces byte-identical training
data.

`csbot.train.sft._supported_kwargs` filters config kwargs to what the installed TRL/transformers
accept and **logs what it dropped**, so a silent API change across these fast-moving libraries
surfaces in the log rather than silently altering a run.

---

## What's real vs. cut for time

### Real, and verifiable from this repo

- **Leakage-free splits.** Cluster-disjoint, not row-random, with the clustering threshold chosen
  by measurement. `scripts/verify_splits.py` re-derives the claim from the data.
- **A model-selection funnel, not a default.** 24 candidates through licence, hardware, zero-shot
  and pilot gates. The hardware gate was wrong at first and eliminated the eventual winner; that
  is recorded rather than quietly fixed.
- **Base-vs-tuned on identical prompts**, greedy, same order, every metric with a bootstrap CI and
  a win/tie/loss count. Regressions surfaced rather than averaged away.
- **A pre-registered behavioural probe suite** that caught a real regression the format metrics
  scored 1.000.
- **Two guardrail mitigations measured separately**, including the finding that the first runtime
  version was a net negative.
- **Serving that matches training byte-for-byte**, with the rendered prompt published at
  `/model-info` so it can be checked rather than trusted.
- **125 tests**, including an integration test of the served stack that asserts a railed request
  performs no generation.
- **Human evaluation.** 20 blind, position-swapped pairs scored on three dimensions — 60
  judgments, with four constructed controls. It is what found the placeholder leak that
  fourteen automatic metrics missed, and it is the reference the LLM judge was calibrated
  against.
- **An LLM judge that had to earn its verdicts.** Three candidates, none from the model's own
  family, against a gate fixed before any ran: control accuracy, position-flip rate, unparsed
  rate, and agreement with the human labels. One failed — Phi-4-mini flipped on 52% of pairs
  when A and B were swapped and agreed with the human 7% of the time. Its verdicts were
  discarded rather than averaged in.

### Cut, and why

- **Human evaluation at scale.** One annotator, 20 pairs, 60 judgments. Enough to find a defect
  and calibrate a judge; not enough for inter-annotator agreement, and not a sample from which
  to generalise about overall quality. Every preference figure in this report rests on that
  single labeller plus one gated model agreeing with them.
- **A judge validated on more than 20 pairs.** Agreement of 0.736 is computed against 60
  judgments. That is a small reference set for a gate that decides which judges are believed.
- **No multi-turn.** The dataset is single-turn, so the model, the template and the evaluation are
  all single-turn. A real deployment needs conversation state, and nothing here measures it.
- **No RAG or tool use.** The model cannot look up an order, which is why so much of the guardrail
  work is about *not* pretending it can. A production system would give it real tools and the
  refusal behaviour would change completely.
- **One seed per configuration.** Seed variance is unmeasured. The paired bootstrap covers
  sampling noise in the *evaluation*, not run-to-run variance in *training*. With a second GPU-day
  this is the first thing to add.
- **`group_by_length` not applied.** Identified as the available throughput win and deliberately
  not used in the final run, because changing two things at once would make the run's behaviour
  no longer attributable to the guardrail data.

### Known to be broken

- **Romanised Hindi is still railed.** Both embedding models are trained on native scripts.
- **Long emotional messages** are diverted 2 times in 4 on the probe set.
- **`policy_pressure` rail recall is 6/20.** The data fix carries most of that load.
- **The intent classifier used as an evaluation metric is itself a model**, with its own ceiling
  reported alongside the numbers it produces. It is a proxy for intent preservation, not ground
  truth.
