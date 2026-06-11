"""Public Voice Activity Detection API (ONNX, torch-free).

Reuses the pyannote/segmentation-3.0 ONNX session. Computes the speaker-agnostic
"anyone speaking" probability per frame (powerset -> multilabel max, averaged
across overlapping windows), hysteresis-binarizes, drops short regions and
bridges short gaps. Returns a pyannote.core.Annotation of "SPEECH" regions.
"""

from __future__ import annotations

import numpy as np
from pyannote.core import Annotation, Segment

from pyannote_onnx_community._lib import (
    hysteresis_binarize as _hysteresis_binarize,
    iter_windows,
    load_segmentation_session,
    resolve_onnx_path,
)
from pyannote_onnx_community.audio import load_audio
from pyannote_onnx_community.config import VADConfig

SEGMENTATION_REPO = "onnx-community/pyannote-segmentation-3.0"

__all__ = ["ONNXVoiceActivityDetection"]


# Sliding-window params — match SD ONNX path + upstream pyannote
# segmentation_step=0.1 (10% of 5s window).
_WINDOW_DURATION = 5.0
_WINDOW_STEP = 0.5


def _segments_from_active_mask(
    active: np.ndarray,
    frame_duration: float,
    *,
    min_duration_on: float,
    min_duration_off: float,
) -> list[tuple[float, float]]:
    """Extract speech intervals from a boolean per-frame active mask.

    Steps:
      1. Walk the mask to extract contiguous (start_sec, end_sec) runs.
      2. Merge consecutive runs whose gap < ``min_duration_off`` (so a brief
         dip below ``offset`` doesn't split a sentence).
      3. Drop runs shorter than ``min_duration_on``.
    """
    if active.size == 0 or not active.any():
        return []

    raw_runs: list[tuple[float, float]] = []
    in_run = False
    start_idx = 0
    for i, flag in enumerate(active):
        if flag and not in_run:
            start_idx = i
            in_run = True
        elif not flag and in_run:
            raw_runs.append((start_idx * frame_duration, i * frame_duration))
            in_run = False
    if in_run:
        raw_runs.append((start_idx * frame_duration, active.size * frame_duration))

    # Merge close runs (gap < min_duration_off).
    if min_duration_off > 0 and raw_runs:
        merged: list[list[float]] = [list(raw_runs[0])]
        for start, end in raw_runs[1:]:
            if start - merged[-1][1] < min_duration_off:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        raw_runs = [(s, e) for s, e in merged]

    # Filter short runs.
    if min_duration_on > 0:
        raw_runs = [(s, e) for s, e in raw_runs if (e - s) >= min_duration_on]

    return raw_runs


_BATCH_SIZE = 32  # matches pyannote.audio Inference default


def _stitched_speech_probability(
    audio: np.ndarray,
    *,
    sample_rate: int,
    session,
    window_duration: float,
    window_step: float,
    batch_size: int = _BATCH_SIZE,
) -> tuple[np.ndarray, float]:
    """Return ``(speech_prob, frame_duration)`` averaged across overlapping windows.

    For each window, run the segmentation session, softmax over the 7-class
    powerset axis, then powerset → multilabel: per local speaker, sum the
    probabilities of classes where that speaker is active (1=A, 4=A+B, 5=A+C
    for speaker A, etc.). ``speech_prob = max(multilabel_A, multilabel_B,
    multilabel_C)`` per frame — matches upstream
    ``VoiceActivityDetection``'s ``pre_aggregation_hook = np.max(scores,
    axis=-1)`` which operates on the multilabel-converted scores
    (Inference.py:126-141 + voice_activity_detection.py:111-113).

    The naive ``1 - softmax[..., 0]`` formulation double-counts overlap
    classes (e.g. softmax[A+B] contributes to both speakers' marginals when
    summed) and gives a smoother speech_prob that biases toward merging
    adjacent speech regions. Use the upstream max-multilabel formula for
    parity.

    Unlike SD, this scalar is speaker-agnostic (max collapses local IDs)
    so averaging across overlapping windows on the absolute frame grid is
    well-defined — chunk-local speaker permutation is a non-issue.

    ``speech_prob`` shape ``(total_frames,)``; ``frame_duration`` is
    ``window_duration / frames_per_window``. Empty audio yields
    ``(empty array, 0.0)``.
    """
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), 0.0

    accum: np.ndarray | None = None
    counts: np.ndarray | None = None
    frames_per_window = 0
    frame_duration = 0.0
    audio_duration = audio.size / sample_rate

    def _flush(batch_audio: list[np.ndarray], batch_offsets: list[float]) -> None:
        """Run one batched session.run and accumulate speech_prob into accum/counts."""
        nonlocal accum, counts, frames_per_window, frame_duration
        if not batch_audio:
            return
        # Stack chunks into (B, 1, samples). All chunks have same length because
        # iter_windows zero-pads the tail; verified by assert below.
        batch = np.stack(batch_audio, axis=0)[:, np.newaxis, :].astype(np.float32)
        out = session.run(None, {"input_values": batch})[0]  # (B, frames, classes)
        # Softmax + powerset → multilabel max in one shot, vectorised across batch.
        shifted = out - out.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
        # Classes: 0=non-speech, 1=A, 2=B, 3=C, 4=A+B, 5=A+C, 6=B+C.
        ml_a = probs[..., 1] + probs[..., 4] + probs[..., 5]
        ml_b = probs[..., 2] + probs[..., 4] + probs[..., 6]
        ml_c = probs[..., 3] + probs[..., 5] + probs[..., 6]
        speech = np.maximum(np.maximum(ml_a, ml_b), ml_c).astype(np.float32)  # (B, frames)

        if accum is None:
            frames_per_window = speech.shape[1]
            frame_duration = window_duration / frames_per_window
            # Clamp the grid to the real audio duration (>=1 frame). Do NOT pad
            # out to a full window: for sub-window clips that would let the
            # trailing-trim below report speech past the true audio end (P1).
            total_frames = max(round(audio_duration / frame_duration), 1)
            accum = np.zeros(total_frames, dtype=np.float32)
            counts = np.zeros(total_frames, dtype=np.int32)

        for chunk_speech, offset_sec in zip(speech, batch_offsets, strict=False):
            start_frame = round(offset_sec / frame_duration)
            end_frame = min(start_frame + frames_per_window, accum.size)
            n = end_frame - start_frame
            if n <= 0:
                continue
            accum[start_frame:end_frame] += chunk_speech[:n]
            counts[start_frame:end_frame] += 1

    batch_audio: list[np.ndarray] = []
    batch_offsets: list[float] = []
    for offset_sec, chunk_audio in iter_windows(
        audio,
        sample_rate=sample_rate,
        window_duration=window_duration,
        window_step=window_step,
    ):
        batch_audio.append(chunk_audio)
        batch_offsets.append(offset_sec)
        if len(batch_audio) >= batch_size:
            _flush(batch_audio, batch_offsets)
            batch_audio = []
            batch_offsets = []
    _flush(batch_audio, batch_offsets)

    if accum is None or counts is None:
        return np.zeros(0, dtype=np.float32), 0.0

    # Trim trailing zero-count frames (audio shorter than the rounded total).
    last_covered = int(np.max(np.nonzero(counts)[0])) + 1 if counts.any() else 0
    accum = accum[:last_covered]
    counts = counts[:last_covered]

    speech_prob = np.zeros_like(accum, dtype=np.float32)
    nonzero = counts > 0
    speech_prob[nonzero] = accum[nonzero] / counts[nonzero]
    return speech_prob, frame_duration


class ONNXVoiceActivityDetection:
    """Torch-free pyannote VAD over ONNX Runtime."""

    def __init__(
        self,
        segmentation_repo: str = SEGMENTATION_REPO,
        config: VADConfig | None = None,
        seg_session=None,
        providers: list[str] | None = None,
    ):
        self.config = config or VADConfig()
        self._providers = providers or ["CPUExecutionProvider"]
        self._seg = seg_session or load_segmentation_session(
            resolve_onnx_path(segmentation_repo, "onnx/model.onnx"), providers=self._providers
        )

    def __call__(self, audio) -> Annotation:
        wav = load_audio(audio, sample_rate=self.config.sample_rate)
        ann = Annotation()
        if wav.size == 0:
            return ann
        speech_prob, frame_duration = _stitched_speech_probability(
            wav,
            sample_rate=self.config.sample_rate,
            session=self._seg,
            window_duration=_WINDOW_DURATION,
            window_step=_WINDOW_STEP,
        )
        if speech_prob.size == 0 or frame_duration == 0.0:
            return ann
        active = _hysteresis_binarize(speech_prob, onset=self.config.onset, offset=self.config.offset)
        for start, end in _segments_from_active_mask(
            active,
            frame_duration,
            min_duration_on=self.config.min_duration_on,
            min_duration_off=self.config.min_duration_off,
        ):
            ann[Segment(start, end)] = "SPEECH"
        return ann.support()
