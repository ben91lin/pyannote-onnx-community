"""Public Speaker Diarization API — mirrors samson / pyannote.

    dia = ONNXSpeakerDiarization(providers=["CPUExecutionProvider"])
    annotation = dia("audio.wav")   # -> pyannote.core.Annotation
"""

from __future__ import annotations

import warnings

from pyannote.core import Annotation, Segment

from pyannote_onnx_community._lib import (
    load_embedding_session,
    load_segmentation_session,
    resolve_onnx_path,
)
from pyannote_onnx_community._pipeline import PyannoteOnnxClient
from pyannote_onnx_community.audio import load_audio
from pyannote_onnx_community.config import SDConfig

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

    def __call__(self, audio, num_speakers: int | None = None) -> Annotation:
        if num_speakers is not None:
            warnings.warn(
                "num_speakers is ignored: the community-1 / VBx path determines the "
                "speaker count automatically.",
                stacklevel=2,
            )
        wav = load_audio(audio, sample_rate=self.config.sample_rate)
        segments = self._client(audio_input=wav, sample_rate=self.config.sample_rate)
        return self._to_annotation(segments)

    @staticmethod
    def _to_annotation(segments) -> Annotation:
        ann = Annotation()
        for seg in segments:
            ann[Segment(seg.start, seg.end)] = seg.speaker
        return ann.support()
