"""Per-(chunk, local-speaker) Pyannote ONNX SD pipeline.

Order of operations:
  1. ``run_segmentation`` — sliding-window forward pass; returns one entry per
     window (offset, audio chunk, softmax probabilities) WITHOUT cross-chunk
     stitching. Pyannote 3.0 segmentation emits per-chunk *local* speaker IDs
     that have no cross-chunk meaning — averaging logits across chunks would
     conflate different real speakers under the same local id.
  2. ``binarize_per_chunk`` — for each chunk, hysteresis-binarize the
     per-(local) speaker probability and compute the single-active mask
     (frames where exactly one speaker is active).
  3. ``extract_embeddings_per_chunk_speaker`` — for each (chunk, local
     speaker) extract one wespeaker embedding from mask-conditioned fbank
     features. Mirrors upstream pyannote.audio 4.0 community-1's
     ``get_embeddings`` + ``filter_embeddings`` (min_active_ratio=0.2).
  4. ``cluster_embeddings_vbx`` — PLDA + VBx clustering (unchanged).
  5. ``_assemble_global_timeline`` — map each (chunk, local) → global cluster
     label, paint per-chunk per-global-speaker activity, concatenate and
     extract contiguous runs as final output segments.

Pure-ONNX community-1 / VBx+PLDA speaker-diarization pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from pyannote.core.utils.generators import string_generator

from pyannote_onnx_community._lib import (
    hysteresis_binarize as _hysteresis_binarize,
    iter_windows,
)
from pyannote_onnx_community._vbx import cluster_vbx
from pyannote_onnx_community.config import SDConfig


def get_speaker_string_generator():
    """A, B, C, ... label generator (pyannote.core string_generator)."""
    return string_generator()


@dataclass
class DiarizationSegment:
    """One diarized span. ``speaker`` is an A/B/C label (or None)."""

    id: int
    start: float
    end: float
    speaker: str | None = None

__all__ = [
    "ChunkSegmentation",
    "PyannoteOnnxClient",
    "assemble_output",
    "binarize_per_chunk",
    "cluster_embeddings_vbx",
    "extract_embeddings_per_chunk_speaker",
    "run_segmentation",
]


# pyannote/segmentation-3.0 emits 7 classes:
#   0: non-speech
#   1, 2, 3: single-speaker active
#   4: speakers 1+2; 5: speakers 1+3; 6: speakers 2+3
NUM_LOCAL_SPEAKERS = 3
_OVERLAP_MEMBERSHIP: dict[int, tuple[int, ...]] = {
    1: (1,),
    2: (2,),
    3: (3,),
    4: (1, 2),
    5: (1, 3),
    6: (2, 3),
}

# Wespeaker (resnet34-LM) minimum frames before the ONNX session starts emitting
# NaNs. Upstream probes this dynamically via ``min_num_samples``; for our use
# (16 kHz, 25 ms / 10 ms fbank, 80 mel bins) the binary search settles at ~25
# frames — hard-code to avoid the probe and make the threshold visible.
_WESPEAKER_MIN_FBANK_FRAMES = 25


@dataclass(frozen=True, slots=True)
class ChunkSegmentation:
    """One sliding-window output: timing + audio + per-frame probabilities."""

    offset_sec: float
    audio: np.ndarray  # (window_samples,) float32 — the chunk waveform
    probs: np.ndarray  # (frames_per_window, 7) float32 — softmax probabilities


# Sliding-window params — match upstream pyannote SpeakerDiarization default
# segmentation_step=0.1 (10% of 5s window). Without overlap (5s/5s), AHC seed
# becomes too sparse and VBx over-prunes. Same params used by VAD provider for
# consistency.
_WINDOW_DURATION = 5.0
_WINDOW_STEP = 0.5
# Powerset binarization thresholds (community-1 yaml: onset == offset == 0.5).
_ONSET = 0.5
_OFFSET = 0.5
_MIN_DURATION_ON = 0.5

_BATCH_SIZE = 32  # matches pyannote.audio Inference default + VAD provider


def run_segmentation(
    *,
    audio: np.ndarray,
    sample_rate: int,
    session,
    window_duration: float,
    window_step: float,
    batch_size: int = _BATCH_SIZE,
) -> tuple[list[ChunkSegmentation], float]:
    """Run sliding-window segmentation, returning per-chunk softmax outputs.

    Returns ``(chunks, frame_duration)``. ``chunks`` is one
    :class:`ChunkSegmentation` per sliding window; ``frame_duration`` is the
    per-frame duration in seconds (``window_duration / frames_per_window``).
    Empty input yields ``([], 0.0)``.

    The session is expected to accept ``input_values`` of shape
    ``(B, 1, window_samples)`` — batch, channel, samples — and emit raw
    logits of shape ``(B, frames_per_window, num_classes=7)``. Logits are
    softmaxed per-frame so downstream binarize comparisons against thresholds
    in ``[0, 1]`` make sense. Default ``batch_size=32`` mirrors
    pyannote.audio.Inference's default; pass ``batch_size=1`` to force
    per-window calls (e.g. when a session can't accept batched input).

    No cross-chunk stitching is performed: pyannote 3.0 segmentation emits
    per-chunk *local* speaker IDs whose mapping to real speakers changes from
    chunk to chunk, so averaging overlapping logits would conflate different
    real speakers. Batching just amortises the ONNX dispatch overhead — each
    chunk's softmax probabilities are still kept separate.
    """
    if audio.size == 0:
        return [], 0.0

    chunks: list[ChunkSegmentation] = []
    frame_duration = 0.0

    def _flush(batch_audio: list[np.ndarray], batch_offsets: list[float]) -> None:
        """Run one batched session.run and append per-chunk softmax probs."""
        nonlocal frame_duration
        if not batch_audio:
            return
        # Stack into (B, 1, samples). iter_windows zero-pads the tail so all
        # chunks have identical length.
        batch = np.stack(batch_audio, axis=0)[:, np.newaxis, :].astype(np.float32)
        out = session.run(None, {"input_values": batch})[0]  # (B, frames, classes)
        # Softmax along class axis, vectorised across batch.
        shifted = out - out.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
        if frame_duration == 0.0:
            frame_duration = window_duration / probs.shape[1]
        for chunk_audio, offset_sec, chunk_probs in zip(batch_audio, batch_offsets, probs, strict=False):
            chunks.append(ChunkSegmentation(offset_sec=offset_sec, audio=chunk_audio, probs=chunk_probs))

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

    return chunks, frame_duration


def _per_speaker_probability(probs: np.ndarray) -> dict[int, np.ndarray]:
    """Compute per-speaker probability timeline as the sum across membership classes.

    Speaker k's marginal probability at frame f = sum of probs[f, c] for all
    classes c whose membership includes speaker k. The powerset classes are
    mutually exclusive (softmax over the 7-class axis), so the speaker marginal
    is the *sum* of the classes that include them — e.g. P(A active) =
    P(A) + P(A+B) + P(A+C). This matches the VAD path (``vad.py`` multilabel
    conversion) and upstream ``pyannote.audio``. Using max here instead would
    systematically under-estimate the marginal when class mass is split across
    a speaker's solo and overlap classes, dropping speakers below threshold.
    """
    contributors: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    for class_idx, members in _OVERLAP_MEMBERSHIP.items():
        for speaker in members:
            contributors[speaker].append(probs[:, class_idx])
    return {speaker: np.add.reduce(arrays).astype(np.float32) for speaker, arrays in contributors.items()}


@dataclass(frozen=True, slots=True)
class ChunkSpeakerMask:
    """Per-(chunk, local speaker) binary masks at segmentation frame rate."""

    chunk_idx: int
    chunk_offset_sec: float
    local_speaker_id: int  # 1, 2, or 3
    binary_mask: np.ndarray  # (frames_per_window,) bool — speaker active
    single_active_mask: np.ndarray  # (frames_per_window,) bool — only one speaker active
    is_overlap_dominant: bool  # informational


def binarize_per_chunk(
    chunks: list[ChunkSegmentation],
    *,
    onset: float,
    offset: float,
) -> list[ChunkSpeakerMask]:
    """Per-chunk per-local-speaker hysteresis binarization.

    For each chunk:
      1. Per-speaker probability via :func:`_per_speaker_probability`.
      2. Hysteresis binarize each local speaker's probability with
         ``onset`` / ``offset`` (set them equal for a hysteresis-free
         per-class threshold — community-1 uses argmax-style decoding so we
         keep the same semantics here).
      3. ``single_active_mask`` = True at frames where exactly one local
         speaker fires (used downstream for overlap-exclusion when extracting
         embeddings — community-1's ``embedding_exclude_overlap=true``).

    Returns one :class:`ChunkSpeakerMask` per (chunk, speaker) where the
    speaker ever fires within the chunk. Speakers that never fire are
    omitted (no embedding to extract).
    """
    out: list[ChunkSpeakerMask] = []
    for chunk_idx, chunk in enumerate(chunks):
        speaker_probs = _per_speaker_probability(chunk.probs)
        # Per-speaker binary masks (frames_per_window, num_local_speakers=3)
        per_speaker_active = {
            speaker: _hysteresis_binarize(prob, onset=onset, offset=offset) for speaker, prob in speaker_probs.items()
        }
        active_stack = np.stack([per_speaker_active[s] for s in (1, 2, 3)], axis=-1)
        single_active_mask = active_stack.sum(axis=-1) == 1
        if chunk.probs.shape[0] > 0:
            argmax_classes = chunk.probs.argmax(axis=1)
        else:
            argmax_classes = np.array([], dtype=np.int64)
        for speaker in (1, 2, 3):
            mask = per_speaker_active[speaker]
            if not mask.any():
                continue
            # Diagnostic: was this local-speaker's activity dominated by overlap classes?
            speaker_frames = argmax_classes[mask]
            is_overlap_dominant = bool(speaker_frames.size > 0 and (speaker_frames >= 4).mean() > 0.5)
            out.append(
                ChunkSpeakerMask(
                    chunk_idx=chunk_idx,
                    chunk_offset_sec=chunk.offset_sec,
                    local_speaker_id=speaker,
                    binary_mask=mask,
                    single_active_mask=single_active_mask,
                    is_overlap_dominant=is_overlap_dominant,
                )
            )
    return out


def _compute_fbank(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute 80-dim Kaldi-style mel filterbank features matching upstream wespeaker.

    Matches WeSpeakerFeatureExtractor preprocessor_config.json:
    frame_length=25ms, frame_shift=10ms, num_mel_bins=80, snip_edges=True,
    dither=0, window_type=hamming, round_to_power_of_two=True.

    Two preprocessing steps mirror pyannote.audio's
    ``ONNXWeSpeakerPretrainedSpeakerEmbedding.compute_fbank``
    (speaker_verification.py:520-566):

      1. ``waveforms = waveforms * (1 << 15)`` — scale float32 [-1, 1] to int16
         range [-32768, 32767]; without this, kaldi fbank energies are 15
         orders of magnitude too small.
      2. ``features - features.mean(dim=time, keepdim=True)`` — per-utterance
         time-mean subtraction; required for the model's expected input
         distribution.

    Without these, embeddings are nearly homogeneous (cosine ~0.5 between
    different speakers); PLDA scoring degenerates and VBx collapses all
    speakers into 1-2 clusters. Verified against upstream on 5 chunks of
    test_audio_47.m4a: per-chunk cosine = 1.0000.

    Returns shape (T, 80) float32.
    """
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_rate)
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.snip_edges = True
    opts.frame_opts.round_to_power_of_two = True
    opts.mel_opts.num_bins = 80
    opts.use_energy = False

    fbank = knf.OnlineFbank(opts)
    # Step 1: scale to int16 range (upstream `waveforms * (1 << 15)`).
    fbank.accept_waveform(float(sample_rate), (audio.astype(np.float32) * 32768.0))
    fbank.input_finished()
    if fbank.num_frames_ready == 0:
        return np.zeros((0, 80), dtype=np.float32)
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]).astype(np.float32)
    # Step 2: per-time mean subtraction (upstream `features - features.mean(dim=1, keepdim=True)`).
    return feats - feats.mean(axis=0, keepdims=True)


def _resample_mask_to(mask: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbour resample a boolean frame mask to ``target_len`` frames.

    Mirrors upstream's ``F.interpolate(..., mode="nearest")`` over the time
    axis (used to align segmentation frames with fbank frames). Empty source
    or target yields an all-False target-length mask.
    """
    if target_len <= 0 or mask.size == 0:
        return np.zeros(target_len, dtype=bool)
    # Sample mask at fractional indices [0, src_len) — same indexing as
    # F.interpolate(mode="nearest") with align_corners=False.
    src_indices = (np.arange(target_len) * (mask.size / target_len)).astype(np.int64)
    src_indices = np.clip(src_indices, 0, mask.size - 1)
    return mask[src_indices]


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    """Provenance for a single extracted embedding."""

    chunk_idx: int
    chunk_offset_sec: float
    local_speaker_id: int
    # Frame mask at SEGMENTATION frame rate (frames_per_window,) — kept so the
    # caller can reconstruct per-frame activity for the (chunk, speaker) when
    # painting the global timeline.
    frame_mask: np.ndarray


def extract_embeddings_per_chunk_speaker(
    *,
    sample_rate: int,
    session,
    chunks: list[ChunkSegmentation],
    speaker_masks: list[ChunkSpeakerMask],
    embedding_exclude_overlap: bool,
    min_active_ratio: float = 0.2,
) -> tuple[np.ndarray, list[EmbeddingMetadata]]:
    """For each (chunk, local-speaker) extract one mask-conditioned embedding.

    Steps per (chunk, speaker):
      1. Compute fbank ONCE per chunk (cached across speakers).
      2. Choose the conditioning mask:
         * ``embedding_exclude_overlap=True`` →
           ``binary_mask & single_active_mask`` (drop overlap-polluted frames);
         * else → ``binary_mask`` (use all of the speaker's frames).
      3. Skip if the speaker is active for less than ``min_active_ratio`` of
         the chunk frames (matches upstream ``filter_embeddings``).
      4. Nearest-neighbour resample the mask from segmentation frame rate to
         fbank frame rate (segmentation ~58 frames / 5 s vs fbank 500 / 5 s).
      5. ``masked_fbank = fbank[interp_mask]``; skip if too short for
         wespeaker (< ``_WESPEAKER_MIN_FBANK_FRAMES``).
      6. Run wespeaker ONNX → 256-D L2-normalised embedding.

    Returns ``(embeddings, metadata)`` — embeddings shape ``(N, 256)`` and
    one :class:`EmbeddingMetadata` per row.
    """
    embedding_dim = 256  # wespeaker-voxceleb-resnet34-LM
    if not speaker_masks:
        return np.zeros((0, embedding_dim), dtype=np.float32), []

    # Cache fbank per chunk (multiple speakers may share a chunk)
    fbank_cache: dict[int, np.ndarray] = {}

    embeddings: list[np.ndarray] = []
    metadata: list[EmbeddingMetadata] = []
    for sm in speaker_masks:
        num_frames = sm.binary_mask.size
        if num_frames == 0:
            continue
        if embedding_exclude_overlap:
            conditioning_mask = sm.binary_mask & sm.single_active_mask
        else:
            conditioning_mask = sm.binary_mask
        # min_active_ratio filter (mirrors upstream filter_embeddings).
        # Upstream measures clean-frame ratio against the chunk's total frame
        # count regardless of exclude_overlap; we use the conditioning mask
        # which collapses to the binary_mask when exclude_overlap=False.
        if conditioning_mask.sum() < min_active_ratio * num_frames:
            continue

        chunk = chunks[sm.chunk_idx]
        if sm.chunk_idx not in fbank_cache:
            fbank_cache[sm.chunk_idx] = _compute_fbank(chunk.audio, sample_rate)
        fbank = fbank_cache[sm.chunk_idx]
        if fbank.shape[0] == 0:
            continue

        interp_mask = _resample_mask_to(conditioning_mask, fbank.shape[0])
        masked_fbank = fbank[interp_mask]
        if masked_fbank.shape[0] < _WESPEAKER_MIN_FBANK_FRAMES:
            continue

        ort_inputs = {"input_features": masked_fbank[np.newaxis, :, :]}  # (1, T', 80)
        out = session.run(None, ort_inputs)[0]  # (1, 256)
        emb = out[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        embeddings.append(emb)
        metadata.append(
            EmbeddingMetadata(
                chunk_idx=sm.chunk_idx,
                chunk_offset_sec=sm.chunk_offset_sec,
                local_speaker_id=sm.local_speaker_id,
                frame_mask=sm.binary_mask.copy(),
            )
        )

    if not embeddings:
        return np.zeros((0, embedding_dim), dtype=np.float32), []
    return np.stack(embeddings).astype(np.float32), metadata


def cluster_embeddings_vbx(
    embeddings: np.ndarray,
    *,
    plda,
    threshold: float,
    fa: float,
    fb: float,
    max_iters: int = 20,
) -> np.ndarray:
    """PLDA + VBx clustering (pyannote.audio 4.0 community-1 algorithm).

    Mirrors the reference implementation in
    ``pyannote.audio.pipelines.clustering.VBxClustering.__call__``:

      1. L2-normalise embeddings.
      2. Hierarchical AHC via ``scipy.cluster.hierarchy.linkage`` with
         ``method="centroid"``, ``metric="euclidean"`` on normed embeddings.
      3. ``fcluster(threshold, criterion="distance")`` to seed initial clusters.
      4. ``plda(embeddings)`` projects raw embeddings into the PLDA latent space.
      5. ``cluster_vbx`` runs the variational Bayes refinement; returns
         ``(gamma, pi)`` where ``gamma`` is responsibilities and ``pi`` are
         learned speaker priors.
      6. Drop speakers whose ``pi <= 1e-7`` (VBx-pruned), then assign each
         embedding to ``argmax`` of remaining responsibilities.

    Args:
        embeddings: ``(N, D)`` raw wespeaker embeddings (already L2-normalised
            by extraction, but we re-normalise for safety).
        plda: A loaded :class:`pyannote_onnx_community._plda.PLDA`
            instance. Provides ``plda(embeddings)`` and ``plda.phi``.
        threshold: AHC distance threshold for the seed clustering.
        fa: VBx ``Fa`` hyper-parameter (community-1 default ``0.07``).
        fb: VBx ``Fb`` hyper-parameter (community-1 default ``0.8``).
        max_iters: VBx iteration cap (upstream uses 20).

    Returns:
        ``(N,)`` int64 array of cluster labels. Empty input returns empty
        array; single input returns ``[0]``.
    """
    if embeddings.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if embeddings.shape[0] == 1:
        return np.array([0], dtype=np.int64)

    logger = logging.getLogger("pyannote_onnx_sd")

    # AHC on L2-normalised embeddings (matches upstream VBxClustering)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dendrogram = linkage(normed, method="centroid", metric="euclidean")
    ahc_clusters = fcluster(dendrogram, threshold, criterion="distance") - 1
    _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)

    # PLDA-project then VBx
    fea = plda(embeddings)
    gamma, pi = cluster_vbx(ahc_clusters, fea, plda.phi, Fa=fa, Fb=fb, maxIters=max_iters)

    # Keep only speakers VBx didn't prune (pi > 1e-7) — same threshold as upstream
    keep = pi > 1e-7
    if not keep.any():
        # Degenerate: fall back to AHC seed labels
        return ahc_clusters.astype(np.int64)

    kept_indices = np.where(keep)[0]
    labels_in_kept = gamma[:, keep].argmax(axis=1)
    labels = kept_indices[labels_in_kept].astype(np.int64)

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "cluster_vbx: ahc=%d → vbx=%d speakers (n=%d, threshold=%.2f, Fa=%.3f, Fb=%.3f)",
            len(np.unique(ahc_clusters)),
            int(keep.sum()),
            embeddings.shape[0],
            threshold,
            fa,
            fb,
        )

    return labels


_MERGE_GAP_SECONDS = 0.5


def _build_speaker_mapping(num_speakers: int) -> dict[int, str]:
    """Map cluster id → A, B, C, ... using the standard generator."""
    gen = get_speaker_string_generator()
    return {i: next(gen) for i in range(num_speakers)}


def assemble_output(
    segments_with_clusters: list[tuple[float, float, int]],
) -> list[DiarizationSegment]:
    """Group, merge, map clusters to speaker labels, return DiarizationSegment list.

    Args:
        segments_with_clusters: ``(start_seconds, end_seconds, cluster_id)``.

    Behavior:
      - Sort by start time.
      - Merge consecutive segments sharing a cluster id when gap < 0.5s.
      - Assign sequential SPEAKER_NN labels to clusters in order of first
        appearance using ``get_speaker_string_generator`` (A, B, C, ...).
    """
    if not segments_with_clusters:
        return []

    sorted_segs = sorted(segments_with_clusters, key=lambda s: s[0])

    merged: list[list[float | int]] = []
    for start, end, cluster in sorted_segs:
        if merged and merged[-1][2] == cluster and (start - merged[-1][1]) < _MERGE_GAP_SECONDS:
            merged[-1][1] = end
        else:
            merged.append([start, end, cluster])

    # Map cluster ids to speakers in first-appearance order
    seen_clusters: list[int] = []
    for _start, _end, cluster in merged:
        if cluster not in seen_clusters:
            seen_clusters.append(cluster)
    cluster_to_speaker = dict(zip(seen_clusters, _build_speaker_mapping(len(seen_clusters)).values(), strict=False))

    return [
        DiarizationSegment(
            id=idx,
            start=float(start),
            end=float(end),
            speaker=cluster_to_speaker[int(cluster)],
        )
        for idx, (start, end, cluster) in enumerate(merged)
    ]


def _runs_from_active_mask(
    active: np.ndarray,
    frame_offsets_sec: np.ndarray,
    frame_duration: float,
) -> list[tuple[float, float]]:
    """Extract (start_sec, end_sec) for contiguous runs of True in ``active``."""
    if not active.any():
        return []
    runs: list[tuple[float, float]] = []
    in_run = False
    start_idx = 0
    for i, flag in enumerate(active):
        if flag and not in_run:
            start_idx = i
            in_run = True
        elif not flag and in_run:
            runs.append((float(frame_offsets_sec[start_idx]), float(frame_offsets_sec[i - 1] + frame_duration)))
            in_run = False
    if in_run:
        runs.append((float(frame_offsets_sec[start_idx]), float(frame_offsets_sec[-1] + frame_duration)))
    return runs


def _assemble_global_timeline(
    *,
    chunks: list[ChunkSegmentation],
    metadata: list[EmbeddingMetadata],
    labels: np.ndarray,
    frame_duration: float,
    min_duration_on: float,
    audio_duration: float,
) -> list[DiarizationSegment]:
    """Paint per-chunk per-global-speaker activity, then extract segments.

    For each (chunk, local speaker) we have a global cluster label from the
    VBx output. Build a per-chunk activity tensor of shape
    ``(num_global_speakers, frames_per_window)`` by OR-ing in each local
    speaker's binary mask under their assigned global label, then walk the
    timeline to extract contiguous runs per global speaker.

    Frames covered by multiple chunks (via overlapping windows) are OR-ed
    together at the absolute frame grid — preserving any chunk that fired the
    speaker. Runs shorter than ``min_duration_on`` are dropped.
    """
    if not metadata or not chunks:
        return []

    # Group metadata by chunk for efficient lookup
    by_chunk: dict[int, list[tuple[EmbeddingMetadata, int]]] = {}
    for meta, label in zip(metadata, labels.tolist(), strict=True):
        by_chunk.setdefault(meta.chunk_idx, []).append((meta, int(label)))

    global_labels = sorted({int(label) for label in labels.tolist()})
    if not global_labels:
        return []
    label_to_row = {label: row for row, label in enumerate(global_labels)}
    num_global = len(global_labels)

    frames_per_window = chunks[0].probs.shape[0]
    last_chunk = chunks[-1]
    # iter_windows zero-pads the final window, so the padded grid can extend
    # past the real audio. Clamp the frame grid back to the true duration so
    # segments never run past the input (P1).
    padded_frames = round((last_chunk.offset_sec + frames_per_window * frame_duration) / frame_duration)
    total_frames = min(padded_frames, round(audio_duration / frame_duration))
    if total_frames <= 0:
        return []
    activity = np.zeros((num_global, total_frames), dtype=bool)

    for chunk_idx, entries in by_chunk.items():
        chunk = chunks[chunk_idx]
        start_frame = round(chunk.offset_sec / frame_duration)
        end_frame = min(start_frame + frames_per_window, total_frames)
        n_frames = end_frame - start_frame
        if n_frames <= 0:
            continue
        for meta, label in entries:
            row = label_to_row[label]
            activity[row, start_frame:end_frame] |= meta.frame_mask[:n_frames]

    frame_offsets_sec = np.arange(total_frames) * frame_duration
    segments_with_clusters: list[tuple[float, float, int]] = []
    for row, label in enumerate(global_labels):
        for start_sec, end_sec in _runs_from_active_mask(activity[row], frame_offsets_sec, frame_duration):
            if (end_sec - start_sec) >= min_duration_on:
                segments_with_clusters.append((start_sec, min(end_sec, audio_duration), label))

    return assemble_output(segments_with_clusters)


@dataclass
class PyannoteOnnxClient:
    """Single-pipeline SD client. Per-(chunk, local speaker) embedding extraction.

    Mirrors pyannote.audio 4.0 community-1: each chunk independently produces
    one embedding per active local speaker (mask-conditioned wespeaker), then
    VBx clustering + global timeline assembly.
    """

    seg_session: object
    emb_session: object
    config: SDConfig
    plda: object | None = None  # lazily loaded on first cluster call

    def _load_plda(self):
        """Lazily resolve the community-1 PLDA artifacts via HF cache."""
        if self.plda is None:
            from pyannote_onnx_community._plda import PLDA

            self.plda = PLDA.from_pretrained(self.config.plda_repo_id, subfolder="plda")
            if self.plda is None:
                raise RuntimeError(
                    f"Failed to resolve PLDA artifacts from {self.config.plda_repo_id!r}. "
                    "Ensure the bundle has been staged (plda/plda.npz + plda/xvec_transform.npz)."
                )
        return self.plda

    def __call__(self, *, audio_input: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        logger = logging.getLogger("pyannote_onnx_sd")

        logger.debug(
            "audio shape=%s dtype=%s duration=%.1fs",
            audio_input.shape,
            audio_input.dtype,
            audio_input.size / sample_rate if audio_input.size else 0,
        )
        if audio_input.size == 0:
            return []

        # 1. Per-chunk segmentation forward pass
        chunks, frame_duration = run_segmentation(
            audio=audio_input,
            sample_rate=sample_rate,
            session=self.seg_session,
            window_duration=_WINDOW_DURATION,
            window_step=_WINDOW_STEP,
        )
        if not chunks:
            return []

        # 2. Per-chunk per-local-speaker binarization + single-active mask
        speaker_masks = binarize_per_chunk(chunks, onset=_ONSET, offset=_OFFSET)
        if not speaker_masks:
            return []

        # 3. Per-(chunk, local) embedding extraction with mask conditioning
        embeddings, metadata = extract_embeddings_per_chunk_speaker(
            sample_rate=sample_rate,
            session=self.emb_session,
            chunks=chunks,
            speaker_masks=speaker_masks,
            embedding_exclude_overlap=self.config.embedding_exclude_overlap,
        )
        if embeddings.shape[0] == 0:
            return []

        # 4. PLDA + VBx clustering (community-1 algorithm)
        plda = self._load_plda()
        labels = cluster_embeddings_vbx(
            embeddings,
            plda=plda,
            threshold=self.config.clustering_threshold,
            fa=self.config.vbx_fa,
            fb=self.config.vbx_fb,
        )

        # 5. Reconstruct timeline by painting per-frame per-global-speaker activity
        out = _assemble_global_timeline(
            chunks=chunks,
            metadata=metadata,
            labels=labels,
            frame_duration=frame_duration,
            min_duration_on=_MIN_DURATION_ON,
            audio_duration=audio_input.size / sample_rate,
        )
        logger.info(
            "pyannote_onnx SD: %d chunks → %d (chunk,speaker) masks → %d embeddings → %d speakers → %d output segments",
            len(chunks),
            len(speaker_masks),
            embeddings.shape[0],
            len(set(labels.tolist())),
            len(out),
        )
        return out
