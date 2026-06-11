"""Torch-free count-based diarization reconstruction.

Numpy re-implementation of upstream pyannote.audio's count-based pipeline
(``SpeakerDiarizationMixin.speaker_count`` / ``reconstruct`` / ``to_diarization``
/ ``to_annotation``), without any ``torch`` / ``pyannote.audio`` dependency.

The three pure stages here operate on plain arrays so they can be unit-tested
in isolation; ``_pipeline.PyannoteOnnxClient`` does the per-(chunk, local
speaker) bookkeeping that feeds them.

Stages:
  1. ``aggregate_overlap_add`` — overlap-add chunk activations onto the
     absolute frame grid (mirrors ``Inference.aggregate(hamming=False)``).
     ``average=True`` divides by per-frame overlap weight (for speaker
     counting); ``average=False`` returns the raw sum (for to_diarization,
     where only per-frame argsort order matters).
  2. ``top_count_discrete`` — per frame, turn on the ``count[t]`` speakers with
     the highest aggregated activation (mirrors ``to_diarization``'s top-k).
  3. ``discrete_to_segments`` — turn a discrete 0/1 timeline into
     ``(start, end, speaker_col)`` spans with ``min_duration_off`` gap-filling
     and ``min_duration_on`` dropping (mirrors ``Binarize`` on 0/1 input).
"""

from __future__ import annotations

import numpy as np


def aggregate_overlap_add(
    per_chunk: list[np.ndarray],
    offsets_frames: list[int],
    *,
    total_frames: int,
    average: bool,
) -> np.ndarray:
    """Overlap-add per-chunk ``(frames, K)`` activations onto the frame grid.

    Args:
        per_chunk: list of ``(frames_per_window, K)`` arrays (same ``K``).
        offsets_frames: start frame on the absolute grid for each chunk.
        total_frames: length of the output grid.
        average: divide each frame by the number of contributing chunks
            (``True``) or return the raw sum (``False``). Frames covered by no
            chunk stay ``0``.

    Returns:
        ``(total_frames, K)`` float64 array.
    """
    k = per_chunk[0].shape[1] if per_chunk else 0
    acc = np.zeros((total_frames, k), dtype=np.float64)
    weight = np.zeros(total_frames, dtype=np.float64)
    for data, start in zip(per_chunk, offsets_frames, strict=True):
        end = min(start + data.shape[0], total_frames)
        n = end - start
        if n <= 0:
            continue
        acc[start:end] += data[:n]
        weight[start:end] += 1.0
    if average:
        nz = weight > 0
        acc[nz] /= weight[nz, None]
    return acc


def top_count_discrete(activations: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Per frame, activate the ``count[t]`` highest-scoring speakers.

    Args:
        activations: ``(total_frames, num_speakers)`` aggregated activations.
        count: ``(total_frames,)`` instantaneous speaker count.

    Returns:
        ``(total_frames, num_speakers)`` 0/1 array.
    """
    _, num_speakers = activations.shape
    if num_speakers == 0:
        return np.zeros_like(activations)
    counts = np.minimum(count.astype(np.int64), num_speakers)
    # rank[t, s] = position of speaker s when speakers are sorted by activation
    # descending at frame t (0 = highest). The top-``count`` speakers are exactly
    # those whose rank is below the frame's count. Vectorised over all frames.
    ranks = np.argsort(np.argsort(-activations, axis=1), axis=1)
    return (ranks < counts[:, None]).astype(activations.dtype)


def reconstruct_from_chunks(
    *,
    soft_per_chunk: list[np.ndarray],
    binary_per_chunk: list[np.ndarray],
    labels_per_chunk: list[np.ndarray],
    offsets_frames: list[int],
    total_frames: int,
    num_global: int,
    frame_duration: float,
    min_duration_on: float,
    min_duration_off: float,
) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]]]:
    """Build overlap-aware and exclusive timelines via count-based reconstruction.

    Mirrors upstream ``reconstruct`` + ``to_diarization`` for both the regular
    (instantaneous ``count``) and exclusive (``count`` clamped to 1) outputs.

    Args:
        soft_per_chunk: per chunk, ``(frames, num_local)`` soft per-local-speaker
            activation (marginal probability).
        binary_per_chunk: per chunk, ``(frames, num_local)`` 0/1 activity — used
            only for instantaneous speaker counting.
        labels_per_chunk: per chunk, ``(num_local,)`` global cluster id for each
            local speaker, or ``-2`` to skip (filtered / inactive).
        offsets_frames: per chunk, start frame on the absolute grid.
        total_frames: length of the absolute frame grid.
        num_global: number of global speaker clusters.
        frame_duration: seconds per frame.
        min_duration_on / min_duration_off: passed to ``discrete_to_segments``.

    Returns:
        ``(speaker_segments, exclusive_segments)`` — each a list of
        ``(start_sec, end_sec, global_cluster_id)``.
    """
    if total_frames <= 0 or num_global == 0:
        return [], []

    # Per chunk: cluster-max the soft activation into (frames, num_global).
    clustered_per_chunk: list[np.ndarray] = []
    count_sums_per_chunk: list[np.ndarray] = []
    for soft, binary, labels in zip(soft_per_chunk, binary_per_chunk, labels_per_chunk, strict=True):
        frames = soft.shape[0]
        clustered = np.zeros((frames, num_global), dtype=np.float64)
        for local, label in enumerate(labels.tolist()):
            if label < 0:
                continue
            clustered[:, label] = np.maximum(clustered[:, label], soft[:, local])
        clustered_per_chunk.append(clustered)
        count_sums_per_chunk.append(binary.sum(axis=1, keepdims=True).astype(np.float64))

    activations = aggregate_overlap_add(
        clustered_per_chunk, offsets_frames, total_frames=total_frames, average=False
    )
    count_real = aggregate_overlap_add(
        count_sums_per_chunk, offsets_frames, total_frames=total_frames, average=True
    )
    count = np.rint(count_real[:, 0]).astype(np.int64)

    speaker_discrete = top_count_discrete(activations, count)
    exclusive_discrete = top_count_discrete(activations, np.minimum(count, 1))

    speaker_segments = discrete_to_segments(
        speaker_discrete, frame_duration, min_duration_on=min_duration_on, min_duration_off=min_duration_off
    )
    exclusive_segments = discrete_to_segments(
        exclusive_discrete, frame_duration, min_duration_on=min_duration_on, min_duration_off=min_duration_off
    )
    return speaker_segments, exclusive_segments


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` frame index runs of True in ``active``."""
    if not active.any():
        return []
    padded = np.concatenate(([False], active, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist(), strict=True))


def discrete_to_segments(
    discrete: np.ndarray,
    frame_duration: float,
    *,
    min_duration_on: float,
    min_duration_off: float,
) -> list[tuple[float, float, int]]:
    """Convert a discrete 0/1 timeline to ``(start_sec, end_sec, col)`` spans.

    Per speaker column: extract active runs, merge runs separated by a gap
    shorter than ``min_duration_off``, then drop runs shorter than
    ``min_duration_on``. Mirrors ``Binarize(onset=offset=0.5)`` on 0/1 input.
    """
    segments: list[tuple[float, float, int]] = []
    for col in range(discrete.shape[1]):
        runs = _runs(discrete[:, col] >= 0.5)
        if not runs:
            continue
        # Merge runs whose intervening off-gap is shorter than min_duration_off.
        merged: list[list[int]] = [list(runs[0])]
        for start, end in runs[1:]:
            gap = (start - merged[-1][1]) * frame_duration
            if gap < min_duration_off:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        for start, end in merged:
            start_sec = start * frame_duration
            end_sec = end * frame_duration
            if (end_sec - start_sec) >= min_duration_on:
                segments.append((start_sec, end_sec, col))
    return segments
