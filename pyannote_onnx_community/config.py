"""Plain-dataclass config for the ONNX diarization / VAD pipelines.

Defaults track pyannote/speaker-diarization-community-1's pipeline yaml.
The AHC seed threshold is bumped from yaml's 0.6 to 0.7 — per-chunk extraction
produces more candidate embeddings than upstream's batched pipeline, so 0.6
gives too fine an AHC seed and VBx over-prunes (validated against the PyTorch
baseline).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _freeze_provider(provider):
    if isinstance(provider, str):
        return provider
    name, options = provider
    return (name, tuple(sorted(dict(options).items())))


@dataclass(frozen=True)
class SessionConfig:
    """ONNX Runtime session settings. Hashable, so it can key the session cache.

    ``providers`` accepts provider names and ``(name, options)`` pairs, the same
    forms ``onnxruntime.InferenceSession`` accepts. ``intra_op_threads=0`` keeps
    the ONNX Runtime default, which sizes the pool from the host.
    """

    providers: tuple = ("CPUExecutionProvider",)
    intra_op_threads: int = 0  # 0 = ONNX Runtime default

    def __post_init__(self):
        if isinstance(self.providers, str):
            raise ValueError("providers must be a sequence, not a single string.")
        frozen = tuple(_freeze_provider(p) for p in self.providers)
        if not frozen:
            raise ValueError("providers must not be empty.")
        object.__setattr__(self, "providers", frozen)
        threads = self.intra_op_threads
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
            raise ValueError(f"intra_op_threads must be an integer >= 0, got {threads!r}.")

    def ort_providers(self) -> list:
        """Providers in the form ``onnxruntime.InferenceSession`` expects."""
        return [p if isinstance(p, str) else (p[0], dict(p[1])) for p in self.providers]

    @classmethod
    def from_env(cls, prefix: str, *, providers=None) -> SessionConfig:
        """Build a config from ``{prefix}_INTRA_OP_THREADS``.

        Unset or blank means ``0``. An invalid value raises ``ValueError`` so a
        bad deployment fails at startup.
        """
        name = f"{prefix}_INTRA_OP_THREADS"
        raw = os.environ.get(name, "").strip()
        try:
            threads = int(raw) if raw else 0
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
        kwargs = {"intra_op_threads": threads}
        if providers:
            kwargs["providers"] = tuple(providers)
        return cls(**kwargs)


@dataclass
class SDConfig:
    """Speaker-diarization hyperparameters (community-1 / VBx+PLDA)."""

    sample_rate: int = 16000
    clustering_threshold: float = 0.7
    vbx_fa: float = 0.07
    vbx_fb: float = 0.8
    plda_repo_id: str = "pyannote/speaker-diarization-community-1"
    embedding_exclude_overlap: bool = True
    # Min-duration filters for count-based reconstruction → Annotation.
    # community-1 yaml: to_annotation(min_duration_on=0.0) and
    # segmentation.min_duration_off=0.0.
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0


@dataclass
class VADConfig:
    """Voice-activity-detection hyperparameters (asr_bench YouTube 'D' config)."""

    sample_rate: int = 16000
    onset: float = 0.5
    offset: float = 0.363
    min_duration_on: float = 2.0
    min_duration_off: float = 1.5
