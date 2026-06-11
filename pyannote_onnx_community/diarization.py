"""Public Speaker Diarization API — mirrors pyannote.audio SpeakerDiarization.

    dia = ONNXSpeakerDiarization(providers=["CPUExecutionProvider"])
    out = dia("audio.wav")                    # -> DiarizeOutput
    out.speaker_diarization                   # pyannote.core.Annotation
    out.exclusive_speaker_diarization         # Annotation, no overlap (for ASR)
    out.speaker_embeddings                    # (num_speakers, dim) array
    out.serialize()                           # JSON-friendly dict
"""

from __future__ import annotations

from collections.abc import Callable

from pyannote.core import Annotation, Segment

from pyannote_onnx_community._lib import (
    load_embedding_session,
    load_segmentation_session,
    resolve_onnx_path,
)
from pyannote_onnx_community._pipeline import PyannoteOnnxClient, SDResult
from pyannote_onnx_community.audio import load_audio
from pyannote_onnx_community.config import SDConfig
from pyannote_onnx_community.output import DiarizeOutput

SEGMENTATION_REPO = "onnx-community/pyannote-segmentation-3.0"
EMBEDDING_REPO = "onnx-community/wespeaker-voxceleb-resnet34-LM"


class ONNXSpeakerDiarization:
    """Torch-free community-1 (VBx + PLDA) diarization over ONNX Runtime."""

    def __init__(
        self,
        segmentation_repo: str = SEGMENTATION_REPO,
        embedding_repo: str = EMBEDDING_REPO,
        config: SDConfig | None = None,
        seg_session=None,
        emb_session=None,
        providers: list[str] | None = None,
    ):
        self.config = config or SDConfig()
        self._providers = providers or ["CPUExecutionProvider"]
        self._seg = seg_session or load_segmentation_session(
            resolve_onnx_path(segmentation_repo, "onnx/model.onnx"), providers=self._providers
        )
        self._emb = emb_session or load_embedding_session(
            resolve_onnx_path(embedding_repo, "onnx/model.onnx"), providers=self._providers
        )
        self._client = PyannoteOnnxClient(seg_session=self._seg, emb_session=self._emb, config=self.config)

    def __call__(
        self,
        audio,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        hook: Callable[..., None] | None = None,
    ) -> DiarizeOutput:
        """Run diarization and return a :class:`DiarizeOutput`.

        Args:
            audio: path or waveform ndarray.
            num_speakers: pin the exact number of speakers.
            min_speakers / max_speakers: bounds; ignored when ``num_speakers``
                is given.
            hook: optional ``hook(step_name, artifact)`` progress callback,
                called at the ``segmentation`` / ``embeddings`` / ``clustering``
                / ``diarization`` stage boundaries.
        """
        wav = load_audio(audio, sample_rate=self.config.sample_rate)
        result = self._client(
            audio_input=wav,
            sample_rate=self.config.sample_rate,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=hook,
        )
        return self._to_output(result)

    @staticmethod
    def _to_output(result: SDResult) -> DiarizeOutput:
        return DiarizeOutput(
            speaker_diarization=_segments_to_annotation(result.speaker),
            exclusive_speaker_diarization=_segments_to_annotation(result.exclusive),
            speaker_embeddings=result.embeddings,
        )


def _segments_to_annotation(segments) -> Annotation:
    ann = Annotation()
    # Unique track per segment so two speakers sharing an identical span in an
    # overlap region are not silently collapsed by the default track key.
    for i, seg in enumerate(segments):
        ann[Segment(seg.start, seg.end), i] = seg.speaker
    return ann.support()
