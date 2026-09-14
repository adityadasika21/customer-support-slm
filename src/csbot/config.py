"""Typed configuration objects, loadable from YAML.

Every run is fully described by one config file plus a seed, and every artefact directory gets a
``manifest.json`` recording the resolved config, the git SHA and the resulting metrics. The funnel
compares a dozen runs; without that discipline it becomes impossible to say afterwards which
numbers came from which settings.
"""

from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Construct a dataclass from a dict, rejecting unknown keys.

    Silently ignoring a misspelled key is how a sweep ends up running twelve identical
    configurations, so unknown keys are a hard error.
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)  # type: ignore[call-arg]


def git_sha() -> str:
    """Short SHA of the working tree, or ``"unknown"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@dataclass
class DataConfig:
    """Which rows a run trains on."""

    parquet: Path = Path("data/processed/dataset.parquet")

    variant: str = "substituted"
    """Ablation arm: ``raw`` keeps ``{{Order Number}}``; ``substituted`` replaces
    context-derivable entities with concrete values. See ``csbot.data.placeholders``."""

    scope: str = "final"
    """``research`` excludes the four held-out intents, so the held-out-intent slice measures
    generalisation to unseen intent types. ``final`` trains on all 27 intents -- the model we
    actually ship. See ``csbot.data.prepare.build_splits``."""

    max_train_rows: int | None = None
    """Cap for pilot runs, so every candidate model sees an identical budget."""

    guardrail_examples: int = 0
    """Synthetic "when not to help" examples blended into training.

    The Bitext data contains no examples of declining -- every row is a customer asking something
    in-domain and an agent helpfully assisting. Fine-tuning on it alone teaches "always help",
    which measurably degraded out-of-domain refusal behaviour (probe pass rate 0.82 -> 0.71).
    These examples put the missing behaviour into the distribution. See csbot.data.guardrails."""

    substantive_upsample: int = 1
    """How many extra copies of the substantive training rows to include.

    A human and a validated judge both preferred the base model's answers, and reading the
    transcripts showed why: the fine-tune's replies are thinner (median 468 chars against base's
    762). Rewriting cannot add substance that was never in the target, and filtering to the
    substantive rows leaves 5,934 of 23,453 and guts 14 of 27 intents. Duplicating them keeps
    every intent and shifts the register toward replies that actually walk a customer through
    something. 1 disables it. See csbot.data.quality.is_substantive."""

    max_seq_len: int = 640
    """Chosen from the measured token-length distribution, not guessed.

    Across the Qwen3 / SmolLM2 / Granite tokenizers on 3,000 rendered training examples:
    mean 211-219, p95 351-366, p99 446-485, max 534-580. 640 clears the longest example with
    margin, so nothing is truncated, while leaving VRAM headroom that a 1024 cap would waste on
    padding. Truncation matters beyond wasted compute: a model trained on clipped responses
    learns to stop mid-sentence, which resurfaces later as an unexplained truncation rate.
    """

    def __post_init__(self) -> None:
        if self.variant not in {"raw", "substituted", "grounded", "polished"}:
            raise ValueError(
                "variant must be 'raw', 'substituted', 'grounded' or 'polished', "
                f"got {self.variant!r}"
            )
        if self.scope not in {"research", "final"}:
            raise ValueError(f"scope must be 'research' or 'final', got {self.scope!r}")
        self.parquet = Path(self.parquet)


@dataclass
class LoraConfig_:
    """LoRA hyperparameters.

    Rank governs adapter capacity. We are teaching style, format and domain convention rather
    than new knowledge, so modest ranks suffice; the sweep confirms rather than assumes this.
    ``alpha = 2 * r`` keeps the effective scaling (``alpha / r``) constant as rank varies, so a
    rank sweep does not silently become a learning-rate sweep.
    """

    r: int = 16
    alpha: int | None = None
    dropout: float = 0.05

    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    """All linear projections, not just attention. Consistently outperforms attention-only
    adaptation at equal parameter budget, and the MLP is where much of the stylistic behaviour
    we are targeting lives."""

    def __post_init__(self) -> None:
        if self.alpha is None:
            self.alpha = 2 * self.r
        self.target_modules = tuple(self.target_modules)


@dataclass
class TrainConfig:
    """A complete training run."""

    model_key: str
    """Key into ``csbot.models.registry.CANDIDATES``."""

    run_name: str
    output_dir: Path = Path("artifacts/runs")

    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig_ = field(default_factory=LoraConfig_)

    # -- optimisation -------------------------------------------------------------------
    learning_rate: float = 2e-4
    """~10x a full fine-tuning rate. Only the adapters move, and they start at zero, so the
    effective step on the merged weights is far smaller than this number suggests."""

    epochs: float = 2.0
    max_steps: int = -1
    """Set for pilots to give every candidate an identical step budget; -1 uses ``epochs``."""

    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    """Effective batch = batch x accumulation. 8 GB forces a small micro-batch, so accumulation
    carries the effective batch up to a stable 32."""

    warmup_ratio: float = 0.03
    """Adapters initialise to zero, making the first steps unusually noisy; a short warmup
    avoids an early destabilising update."""

    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # -- precision / memory -------------------------------------------------------------
    bf16: bool = True
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    """QLoRA. Required for the 3-4B candidates on 8 GB; see ``ModelSpec.fit_tier``."""

    # -- logging / evaluation -----------------------------------------------------------
    eval_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 200
    eval_rows: int = 400
    """Validation rows for the loss curve. Capped because validation loss is a monitoring signal
    here, not the selection metric -- generation quality on held-out data is."""

    sample_generations: int = 8
    """Fixed prompts generated at each eval step and written to disk, so training can be
    inspected qualitatively rather than only as a loss curve."""

    seed: int = 17

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if isinstance(self.data, dict):
            self.data = _from_dict(DataConfig, self.data)
        if isinstance(self.lora, dict):
            self.lora = _from_dict(LoraConfig_, self.lora)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def to_dict(self) -> dict:
        """JSON-serialisable view of the config, for the run manifest.

        Recurses through dicts and lists as well as dataclasses: ``dataclasses.asdict`` already
        flattens nested dataclasses into plain dicts but leaves ``Path`` values untouched inside
        them, so a converter that only handled dataclasses would miss exactly those.
        """

        def convert(o: Any) -> Any:
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                return convert(dataclasses.asdict(o))
            if isinstance(o, dict):
                return {str(k): convert(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [convert(v) for v in o]
            if isinstance(o, Path):
                return str(o)
            return o

        return convert(self)


def load_train_config(path: Path | str, **overrides: Any) -> TrainConfig:
    """Load a training config from YAML, applying command-line overrides."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return _from_dict(TrainConfig, data)
