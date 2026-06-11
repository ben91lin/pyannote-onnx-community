"""Public speaker-embedding API: waveform -> 256-d L2-normalised wespeaker vector.

Uses the same kaldi fbank (`*32768` int16 scaling + per-time CMN) and the same
onnx-community/wespeaker-voxceleb-resnet34-LM model as the diarization path, so
embeddings are training-distribution-matched (unlike a librosa float-domain fbank).
"""

from __future__ import annotations

import numpy as np

from pyannote_onnx_community._lib import load_embedding_session, resolve_onnx_path
from pyannote_onnx_community._pipeline import _WESPEAKER_MIN_FBANK_FRAMES, _compute_fbank
from pyannote_onnx_community.audio import load_audio

EMBEDDING_REPO = "onnx-community/wespeaker-voxceleb-resnet34-LM"


class ONNXSpeakerEmbedding:
    """Extract one L2-normalised 256-d embedding from a waveform/path."""

    def __init__(
        self,
        embedding_repo: str = EMBEDDING_REPO,
        emb_session=None,
        providers: list[str] | None = None,
        sample_rate: int = 16000,
    ):
        self._providers = providers or ["CPUExecutionProvider"]
        self.sample_rate = sample_rate
        self._emb = emb_session or load_embedding_session(
            resolve_onnx_path(embedding_repo, "onnx/model.onnx"), providers=self._providers
        )

    def __call__(self, audio) -> np.ndarray:
        """Return a 256-d L2-normalised embedding.

        Raises ``ValueError`` for empty or too-short audio: fewer than
        ``_WESPEAKER_MIN_FBANK_FRAMES`` fbank frames makes the wespeaker ONNX
        session emit NaNs, so reject the input rather than return garbage.
        """
        wav = load_audio(audio, sample_rate=self.sample_rate)
        feats = _compute_fbank(wav, self.sample_rate)
        if feats.shape[0] < _WESPEAKER_MIN_FBANK_FRAMES:
            raise ValueError(
                f"audio too short for embedding: produced {feats.shape[0]} fbank "
                f"frame(s), need >= {_WESPEAKER_MIN_FBANK_FRAMES} (~0.4s of "
                f"{self.sample_rate} Hz audio)"
            )
        out = self._emb.run(None, {"input_features": feats[np.newaxis, :, :].astype(np.float32)})[0][0]
        norm = float(np.linalg.norm(out))
        return (out / norm).astype(np.float32) if norm > 1e-6 else out.astype(np.float32)
