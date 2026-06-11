"""Public diarization output container — mirrors pyannote.audio DiarizeOutput.

    output = ONNXSpeakerDiarization()("audio.wav")
    output.speaker_diarization            # pyannote.core.Annotation (overlap-aware)
    output.exclusive_speaker_diarization  # Annotation, no overlapping turns (for ASR)
    output.speaker_embeddings             # (num_speakers, dim) array, labels() order
    output.serialize()                    # JSON-friendly dict
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyannote.core import Annotation


@dataclass
class DiarizeOutput:
    """Speaker diarization result (mirrors upstream ``DiarizeOutput``)."""

    # Overlap-aware speaker diarization.
    speaker_diarization: Annotation

    # Diarization adapted to downstream transcription — no overlapping turns.
    exclusive_speaker_diarization: Annotation

    # One embedding per speaker, ``(num_speakers, dimension)``, sorted in
    # ``speaker_diarization.labels()`` order. ``None`` when unavailable.
    speaker_embeddings: np.ndarray | None = None

    def serialize(self) -> dict[str, Any]:
        """JSON-friendly dict with ``diarization`` and ``exclusive_diarization``.

        Each entry is ``{"start", "end", "speaker"}`` with times rounded to
        milliseconds — matches upstream ``DiarizeOutput.serialize``.
        """
        return {
            "diarization": _turns(self.speaker_diarization),
            "exclusive_diarization": _turns(self.exclusive_speaker_diarization),
        }


def _turns(annotation: Annotation) -> list[dict[str, Any]]:
    return [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
