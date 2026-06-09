"""Plain-dataclass config for the ONNX diarization / VAD pipelines.

Defaults track pyannote/speaker-diarization-community-1's pipeline yaml.
The AHC seed threshold is bumped from yaml's 0.6 to 0.7 — per-chunk extraction
produces more candidate embeddings than upstream's batched pipeline, so 0.6
gives too fine an AHC seed and VBx over-prunes (validated against the PyTorch
baseline).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SDConfig:
    """Speaker-diarization hyperparameters (community-1 / VBx+PLDA)."""

    sample_rate: int = 16000
    clustering_threshold: float = 0.7
    vbx_fa: float = 0.07
    vbx_fb: float = 0.8
    plda_repo_id: str = "pyannote/speaker-diarization-community-1"
    embedding_exclude_overlap: bool = True


@dataclass
class VADConfig:
    """Voice-activity-detection hyperparameters (asr_bench YouTube 'D' config)."""

    sample_rate: int = 16000
    onset: float = 0.5
    offset: float = 0.363
    min_duration_on: float = 2.0
    min_duration_off: float = 1.5
