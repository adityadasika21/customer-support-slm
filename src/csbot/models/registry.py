"""Candidate model registry and the funnel's mechanical gates.

The funnel has three stages. The first two are deterministic and live here:

    Stage 0  license gate   -- does the license permit unrestricted commercial self-hosting?
    Stage 1  hardware gate  -- can it LoRA-train AND serve in bf16 inside our VRAM budget?
    Stage 2  zero-shot screen  (empirical, see csbot.eval)
    Stage 3  LoRA pilots       (empirical, see csbot.train)

Adding a candidate is one ``ModelSpec`` entry in ``CANDIDATES``.

License facts were read from the Hugging Face Hub API (``card_data.license``) rather than from
memory; ``the registry's own metadata`` re-verifies them against the Hub so the table cannot
silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Bytes per parameter when weights are held in bfloat16.
BF16_BYTES_PER_PARAM = 2

# Headroom that must remain free for the KV cache and CUDA/activation overhead before we are
# willing to call a model "servable" under vLLM. Below this, batch size collapses to ~1 and the
# throughput numbers stop being meaningful.
MIN_KV_CACHE_GB = 1.5

# Fixed overhead for CUDA context, fragmentation and framework allocations.
CUDA_OVERHEAD_GB = 0.9

# LoRA training holds frozen bf16 weights plus activations. With gradient checkpointing the
# activation term is roughly proportional to the weights; adapter grads and Adam states are
# negligible (<1% of params). This multiplier is an empirical rule of thumb, validated by the
# smoke test in tests/ -- it is a screening heuristic, not a guarantee.
LORA_TRAIN_MULTIPLIER = 1.35
LORA_TRAIN_OVERHEAD_GB = 1.6

# NF4 (bitsandbytes 4-bit) weights, including the quantisation constants, land a little above the
# nominal 0.5 bytes/param.
NF4_BYTES_PER_PARAM = 0.55


class FitTier(str, Enum):
    """How a model fits the VRAM budget -- the Stage 1 verdict.

    The distinction matters because it is a real engineering trade-off, not just bookkeeping:
    ``BF16`` models are simple to train and serve, while ``QUANTIZED`` models buy extra capacity
    at the cost of a quantisation step on both ends and some quality loss.
    """

    #: LoRA-trainable and servable in bf16. Simplest path, fewest moving parts.
    BF16 = "bf16"
    #: Needs 4-bit QLoRA to train and AWQ/GPTQ to serve. More capacity, more risk.
    QUANTIZED = "quantized"
    #: Does not fit even quantised.
    NO_FIT = "no_fit"


class LicenseVerdict(str, Enum):
    """Outcome of the Stage 0 license gate."""

    #: OSI-approved, no field-of-use restrictions. Preferred.
    PERMISSIVE = "permissive"
    #: Commercial use allowed but with custom terms, gating, or acceptable-use restrictions.
    RESTRICTED = "restricted"
    #: Commercial use not permitted at all.
    NONCOMMERCIAL = "noncommercial"


@dataclass(frozen=True)
class ModelSpec:
    """Everything the pipeline needs to know about a candidate base model."""

    key: str
    """Short stable identifier used in configs, filenames and report tables."""

    repo_id: str
    family: str
    params_b: float
    """Parameter count in billions, read from the Hub's safetensors metadata."""

    license_id: str
    license_verdict: LicenseVerdict
    license_note: str = ""

    gated: bool = False
    """Requires manual access approval on the Hub -- friction for anyone reproducing this run."""

    supports_system_role: bool = True
    """Some chat templates reject a ``system`` turn and need it folded into the first user turn."""

    thinking_model: bool = False
    """Hybrid-reasoning model. Thinking must be disabled for a support assistant; a customer must
    never see a reasoning trace. See ``chat_template_kwargs`` and the ``think_leakage`` metric."""

    chat_template_kwargs: dict = field(default_factory=dict)
    """Extra kwargs passed to ``tokenizer.apply_chat_template`` (e.g. ``enable_thinking=False``)."""

    notes: str = ""

    # -- derived VRAM estimates -------------------------------------------------------------

    @property
    def weights_gb(self) -> float:
        """Size of the bf16 weights on device."""
        return self.params_b * BF16_BYTES_PER_PARAM

    @property
    def est_train_gb(self) -> float:
        """Estimated peak VRAM for LoRA training with gradient checkpointing."""
        return self.weights_gb * LORA_TRAIN_MULTIPLIER + LORA_TRAIN_OVERHEAD_GB

    @property
    def est_serve_gb(self) -> float:
        """Estimated VRAM to serve in bf16 with a usable KV cache."""
        return self.weights_gb + CUDA_OVERHEAD_GB + MIN_KV_CACHE_GB

    @property
    def weights_nf4_gb(self) -> float:
        """Size of 4-bit NF4-quantised weights on device."""
        return self.params_b * NF4_BYTES_PER_PARAM

    @property
    def est_qlora_train_gb(self) -> float:
        """Estimated peak VRAM for 4-bit QLoRA training with gradient checkpointing."""
        return self.weights_nf4_gb * LORA_TRAIN_MULTIPLIER + LORA_TRAIN_OVERHEAD_GB

    @property
    def est_serve_int4_gb(self) -> float:
        """Estimated VRAM to serve 4-bit quantised weights with a usable KV cache."""
        return self.weights_nf4_gb + CUDA_OVERHEAD_GB + MIN_KV_CACHE_GB

    def trainable_on(self, vram_gb: float) -> bool:
        return self.est_train_gb <= vram_gb

    def servable_bf16_on(self, vram_gb: float) -> bool:
        return self.est_serve_gb <= vram_gb

    def train_tier(self, vram_gb: float) -> FitTier:
        """Precision needed to *train* this model on the given budget."""
        if self.trainable_on(vram_gb):
            return FitTier.BF16
        if self.est_qlora_train_gb <= vram_gb:
            return FitTier.QUANTIZED
        return FitTier.NO_FIT

    def serve_tier(self, vram_gb: float) -> FitTier:
        """Precision needed to *serve* this model on the given budget.

        Deliberately independent of :meth:`train_tier`. Training holds activations and optimiser
        state that inference does not, so a model can require 4-bit QLoRA to train and still serve
        comfortably in bf16 -- granite-3.3-2b is exactly that case (8.4 GB to train, 7.5 GB to
        serve). Collapsing the two into one verdict hid that, and made the model look as though it
        forced a quantised checkpoint on whoever loads it. It does not: the adapter merges into
        the full-precision base and ships as an ordinary bf16 checkpoint.
        """
        if self.servable_bf16_on(vram_gb):
            return FitTier.BF16
        if self.est_serve_int4_gb <= vram_gb:
            return FitTier.QUANTIZED
        return FitTier.NO_FIT

    def fit_tier(self, vram_gb: float) -> FitTier:
        """Stage 1 verdict: the more demanding of the training and serving tiers.

        Conservative by construction -- a model counts as ``QUANTIZED`` if *either* end needs it.
        Use :meth:`train_tier` and :meth:`serve_tier` when the distinction matters, which it does
        for the shipped artefact: what gets loaded is the *serving* tier.
        """
        train, serve = self.train_tier(vram_gb), self.serve_tier(vram_gb)
        if FitTier.NO_FIT in (train, serve):
            return FitTier.NO_FIT
        if FitTier.QUANTIZED in (train, serve):
            return FitTier.QUANTIZED
        return FitTier.BF16

    @property
    def license_ok(self) -> bool:
        """Stage 0 verdict: permissive licenses only.

        We deliberately reject ``RESTRICTED`` as well as ``NONCOMMERCIAL``. We ask for a
        licence permitting commercial self-hosting; custom terms with field-of-use restrictions
        and manual gating are a liability a business would have to run past legal, and gating also
        breaks one-command reproduction.
        """
        return self.license_verdict is LicenseVerdict.PERMISSIVE


#: Every model considered, including the rejects. The rejects are kept deliberately: the funnel
#: write-up needs to show *what was excluded and why*, not just the survivors.
CANDIDATES: tuple[ModelSpec, ...] = (
    # -- Qwen ------------------------------------------------------------------------------
    ModelSpec(
        key="qwen3-0.6b",
        repo_id="Qwen/Qwen3-0.6B",
        family="qwen",
        params_b=0.75,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        thinking_model=True,
        chat_template_kwargs={"enable_thinking": False},
        notes="Smallest viable candidate; useful as a floor for the size/quality curve.",
    ),
    ModelSpec(
        key="qwen3-1.7b",
        repo_id="Qwen/Qwen3-1.7B",
        family="qwen",
        params_b=2.03,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        thinking_model=True,
        chat_template_kwargs={"enable_thinking": False},
    ),
    ModelSpec(
        key="qwen2.5-1.5b",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        family="qwen",
        params_b=1.54,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        notes="Older generation but no thinking-mode handling; a simple, well-trodden baseline.",
    ),
    ModelSpec(
        key="qwen3-4b",
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        family="qwen",
        params_b=4.02,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        notes="Stretch goal: needs QLoRA to train and AWQ/FP8 to serve inside 8 GB.",
    ),
    ModelSpec(
        key="qwen3.5-2b",
        repo_id="Qwen/Qwen3.5-2B",
        family="qwen",
        params_b=2.27,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        thinking_model=True,
        chat_template_kwargs={"enable_thinking": False},
        notes="Newest Qwen generation. Fits bf16 on 8 GB, so no quantisation needed either end.",
    ),
    ModelSpec(
        key="qwen3.5-4b",
        repo_id="Qwen/Qwen3.5-4B",
        family="qwen",
        params_b=4.66,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        thinking_model=True,
        chat_template_kwargs={"enable_thinking": False},
    ),
    ModelSpec(
        key="qwen2.5-3b",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen",
        params_b=3.09,
        license_id="qwen-research",
        license_verdict=LicenseVerdict.NONCOMMERCIAL,
        license_note=(
            "Qwen Research License -- non-commercial. Note the 1.5B sibling IS Apache-2.0; "
            "assuming licence uniformity across a family is a common and expensive mistake."
        ),
    ),
    # -- Mistral ---------------------------------------------------------------------------
    ModelSpec(
        key="mistral-7b-v0.3",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        family="mistral",
        params_b=7.25,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        supports_system_role=False,
        notes="Apache-2.0. Largest model that still fits 8 GB once 4-bit quantised.",
    ),
    ModelSpec(
        key="mistral-nemo",
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        family="mistral",
        params_b=12.25,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        notes=(
            "Passes the licence gate (Apache-2.0) but fails the hardware gate: even at 4-bit the "
            "weights are ~6.7 GB, leaving no room for a KV cache on 8 GB. Kept in the registry so "
            "the funnel records that it was considered and excluded on size, not on licence."
        ),
    ),
    # -- Microsoft Phi ---------------------------------------------------------------------
    ModelSpec(
        key="phi-4-mini",
        repo_id="microsoft/Phi-4-mini-instruct",
        family="phi",
        params_b=3.84,
        license_id="mit",
        license_verdict=LicenseVerdict.PERMISSIVE,
        notes="MIT is the cleanest licence in the set. Stretch goal on size.",
    ),
    ModelSpec(
        key="phi-3.5-mini",
        repo_id="microsoft/Phi-3.5-mini-instruct",
        family="phi",
        params_b=3.82,
        license_id="mit",
        license_verdict=LicenseVerdict.PERMISSIVE,
    ),
    # -- HuggingFaceTB SmolLM --------------------------------------------------------------
    ModelSpec(
        key="smollm3-3b",
        repo_id="HuggingFaceTB/SmolLM3-3B",
        family="smollm",
        params_b=3.08,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        thinking_model=True,
        chat_template_kwargs={"enable_thinking": False},
    ),
    ModelSpec(
        key="smollm2-1.7b",
        repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family="smollm",
        params_b=1.71,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
    ),
    # -- IBM Granite -----------------------------------------------------------------------
    ModelSpec(
        key="granite-3.3-2b",
        repo_id="ibm-granite/granite-3.3-2b-instruct",
        family="granite",
        params_b=2.53,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        notes="IBM trains these explicitly for enterprise assistant use -- plausible fit for support.",
    ),
    ModelSpec(
        key="granite-4.0-micro",
        repo_id="ibm-granite/granite-4.0-micro",
        family="granite",
        params_b=3.40,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
    ),
    # -- AllenAI OLMo ----------------------------------------------------------------------
    ModelSpec(
        key="olmo2-1b",
        repo_id="allenai/OLMo-2-0425-1B-Instruct",
        family="olmo",
        params_b=1.48,
        license_id="apache-2.0",
        license_verdict=LicenseVerdict.PERMISSIVE,
        notes="Fully open training data as well as weights -- the strictest reading of 'open source'.",
    ),
    # -- Rejected at Stage 0 ---------------------------------------------------------------
    ModelSpec(
        key="llama3.2-1b",
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
        params_b=1.24,
        license_id="llama3.2",
        license_verdict=LicenseVerdict.RESTRICTED,
        gated=True,
        license_note=(
            "Llama 3.2 Community License: commercial use permitted under 700M MAU, but carries an "
            "acceptable-use policy, attribution/naming requirements, and manual Hub gating. Not OSI."
        ),
    ),
    ModelSpec(
        key="llama3.2-3b",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        params_b=3.21,
        license_id="llama3.2",
        license_verdict=LicenseVerdict.RESTRICTED,
        gated=True,
        license_note="As Llama-3.2-1B.",
    ),
    ModelSpec(
        key="gemma3-1b",
        repo_id="google/gemma-3-1b-it",
        family="gemma",
        params_b=1.00,
        license_id="gemma",
        license_verdict=LicenseVerdict.RESTRICTED,
        gated=True,
        supports_system_role=False,
        license_note=(
            "Gemma Terms of Use: custom licence with a use policy Google may update unilaterally, "
            "plus manual Hub gating. Not OSI."
        ),
    ),
    ModelSpec(
        key="gemma3-4b",
        repo_id="google/gemma-3-4b-it",
        family="gemma",
        params_b=4.30,
        license_id="gemma",
        license_verdict=LicenseVerdict.RESTRICTED,
        gated=True,
        supports_system_role=False,
        license_note="As gemma-3-1b-it.",
    ),
    ModelSpec(
        key="falcon3-1b",
        repo_id="tiiuae/Falcon3-1B-Instruct",
        family="falcon",
        params_b=1.67,
        license_id="falcon-llm-license",
        license_verdict=LicenseVerdict.RESTRICTED,
        license_note="Falcon LLM License: custom terms with an acceptable-use policy. Not OSI.",
    ),
    ModelSpec(
        key="lfm2-1.2b",
        repo_id="LiquidAI/LFM2-1.2B",
        family="lfm",
        params_b=1.17,
        license_id="lfm1.0",
        license_verdict=LicenseVerdict.RESTRICTED,
        license_note="LFM Open License: free use is revenue-capped, so not unrestricted commercial.",
    ),
    ModelSpec(
        key="ministral-8b",
        repo_id="mistralai/Ministral-8B-Instruct-2410",
        family="mistral",
        params_b=8.02,
        license_id="mrl",
        license_verdict=LicenseVerdict.NONCOMMERCIAL,
        license_note=(
            "Mistral Research License -- non-commercial. Note this is Ministral specifically; "
            "Mistral-7B-v0.3 and Mistral-Nemo are Apache-2.0, so the *family* is not excluded."
        ),
    ),
    ModelSpec(
        key="nemotron-3-nano-4b",
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        family="nemotron",
        params_b=3.97,
        license_id="nvidia-nemotron-open-model-license",
        license_verdict=LicenseVerdict.RESTRICTED,
        license_note=(
            "NVIDIA Open Model License: commercial use permitted, but custom terms with "
            "field-of-use conditions rather than an OSI licence. Ungated, so unlike Llama/Gemma "
            "it does not break one-command reproduction."
        ),
    ),
)

BY_KEY: dict[str, ModelSpec] = {m.key: m for m in CANDIDATES}


def get(key: str) -> ModelSpec:
    """Look up a candidate by its short key, with a helpful error on typos."""
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown model key {key!r}. Known keys: {sorted(BY_KEY)}") from None


def stage0_license_survivors() -> list[ModelSpec]:
    """Candidates whose licence permits unrestricted commercial self-hosting."""
    return [m for m in CANDIDATES if m.license_ok]


def stage1_hardware_survivors(
    vram_gb: float, *, tiers: tuple[FitTier, ...] = (FitTier.BF16, FitTier.QUANTIZED)
) -> list[ModelSpec]:
    """Licence survivors that fit the VRAM budget, in one of the accepted tiers.

    Pass ``tiers=(FitTier.BF16,)`` to restrict the funnel to the simple path.
    """
    return sorted(
        (m for m in stage0_license_survivors() if m.fit_tier(vram_gb) in tiers),
        key=lambda m: m.params_b,
    )


def funnel_table(vram_gb: float) -> list[dict]:
    """Rows describing every candidate's journey through Stages 0 and 1.

    Includes rejects -- the funnel write-up needs to show what was excluded and why.
    """
    rows = []
    for m in sorted(CANDIDATES, key=lambda x: (x.family, x.params_b)):
        tier = m.fit_tier(vram_gb) if m.license_ok else None
        rows.append(
            {
                "key": m.key,
                "repo_id": m.repo_id,
                "family": m.family,
                "params_b": m.params_b,
                "license": m.license_id,
                "license_verdict": m.license_verdict.value,
                "stage0_pass": m.license_ok,
                "stage1_tier": tier.value if tier else None,
                "stage1_pass": bool(tier and tier is not FitTier.NO_FIT),
                "est_train_gb": round(m.est_train_gb, 1),
                "est_serve_gb": round(m.est_serve_gb, 1),
                "est_qlora_train_gb": round(m.est_qlora_train_gb, 1),
                "est_serve_int4_gb": round(m.est_serve_int4_gb, 1),
                "reason": m.license_note or m.notes,
            }
        )
    return rows
