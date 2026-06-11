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
  4. ``cluster_embeddings_vbx`` — PLDA + VBx clustering → labels + centroids,
     with optional num/min/max-speaker KMeans force-count.
  5. ``reconstruct_from_chunks`` (in ``_reconstruct``) — count-based
     reconstruction: overlap-add the soft per-(chunk, local) activations onto
     the absolute frame grid, keep the top-``count`` speakers per frame, and
     emit overlap-aware + exclusive timelines. ``assign_speaker_labels`` then
     maps cluster ids to ``SPEAKER_NN`` and aligns the per-speaker embeddings.

Pure-ONNX community-1 / VBx+PLDA speaker-diarization pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from pyannote_onnx_community._lib import (
    hysteresis_binarize as _hysteresis_binarize,
    iter_windows,
)
from pyannote_onnx_community._reconstruct import reconstruct_from_chunks
from pyannote_onnx_community._vbx import cluster_vbx
from pyannote_onnx_community.config import SDConfig


@dataclass
class DiarizationSegment:
    """One diarized span. ``speaker`` is a ``SPEAKER_NN`` label (or None)."""

    id: int
    start: float
    end: float
    speaker: str | None = None

__all__ = [
    "ChunkSegmentation",
    "PyannoteOnnxClient",
    "SDResult",
    "assign_speaker_labels",
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


def _resolve_cluster_bounds(
    num_clusters: int | None, min_clusters: int | None, max_clusters: int | None, n: int
) -> tuple[int | None, int, int]:
    """Resolve (num, min, max) cluster bounds against ``n`` embeddings.

    Mirrors upstream ``BaseClustering.set_num_clusters``: ``num_clusters`` pins
    both bounds; otherwise min defaults to 1 and max to ``n``. All clamped to
    ``[1, n]``.
    """
    lo = num_clusters or min_clusters or 1
    lo = max(1, min(n, lo))
    hi = num_clusters or max_clusters or n
    hi = max(1, min(n, hi))
    if lo > hi:
        raise ValueError(f"min_clusters ({lo}) must be <= max_clusters ({hi}).")
    if lo == hi:
        num_clusters = lo
    return num_clusters, lo, hi


def _kmeans_force(normed: np.ndarray, embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Force exactly ``k`` clusters via scipy kmeans2 on L2-normed embeddings.

    sklearn-free analogue of upstream's KMeans force-count path. Runs a few
    deterministic ``minit='++'`` restarts, keeps the lowest-distortion result
    that yields ``k`` non-empty clusters, and recomputes centroids as the mean
    of the raw embeddings per cluster. Returns ``(labels, centroids)``.
    """
    from scipy.cluster.vq import kmeans2

    best_labels: np.ndarray | None = None
    best_distortion = np.inf
    for seed in range(5):
        centers, labels = kmeans2(normed, k, minit="++", seed=seed, missing="warn")
        if len(np.unique(labels)) != k:
            continue
        distortion = float(np.sum((normed - centers[labels]) ** 2))
        if distortion < best_distortion:
            best_distortion = distortion
            best_labels = labels
    if best_labels is None:
        # Degenerate fallback: round-robin assign so all k clusters are non-empty.
        best_labels = np.arange(normed.shape[0]) % k
    _, best_labels = np.unique(best_labels, return_inverse=True)
    centroids = np.stack([embeddings[best_labels == c].mean(axis=0) for c in range(k)]).astype(np.float32)
    return best_labels.astype(np.int64), centroids


def cluster_embeddings_vbx(
    embeddings: np.ndarray,
    *,
    plda,
    threshold: float,
    fa: float,
    fb: float,
    max_iters: int = 20,
    num_clusters: int | None = None,
    min_clusters: int | None = None,
    max_clusters: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

        num_clusters: pin the exact speaker count (overrides auto).
        min_clusters / max_clusters: bounds; ignored when ``num_clusters`` set.

    Returns:
        ``(labels, centroids)`` — ``labels`` is ``(N,)`` int64; ``centroids`` is
        ``(num_speakers, D)`` float32 (one raw-embedding centroid per speaker,
        indexed by label). Empty input returns empty arrays; single input
        returns ``([0], embeddings[:1])``.
    """
    embedding_dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
    if embeddings.shape[0] == 0:
        return np.array([], dtype=np.int64), np.zeros((0, embedding_dim), dtype=np.float32)
    if embeddings.shape[0] == 1:
        return np.array([0], dtype=np.int64), embeddings.astype(np.float32).copy()

    logger = logging.getLogger("pyannote_onnx_sd")
    n = embeddings.shape[0]
    num_clusters, min_clusters, max_clusters = _resolve_cluster_bounds(
        num_clusters, min_clusters, max_clusters, n
    )

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
        labels = ahc_clusters.astype(np.int64)
        centroids = _centroids_from_labels(embeddings, labels)
        return labels, centroids

    kept_indices = np.where(keep)[0]
    labels_in_kept = gamma[:, keep].argmax(axis=1)
    labels = kept_indices[labels_in_kept].astype(np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    auto_num = len(np.unique(labels))

    # Force-count via KMeans only when the auto count is out of bounds or pinned
    # (mirrors upstream VBxClustering's optional KMeans re-cluster).
    if auto_num < min_clusters:
        num_clusters = min_clusters
    elif auto_num > max_clusters:
        num_clusters = max_clusters
    if num_clusters and num_clusters != auto_num:
        labels, centroids = _kmeans_force(normed, embeddings, num_clusters)
    else:
        centroids = _centroids_from_labels(embeddings, labels)

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "cluster_vbx: ahc=%d → vbx=%d → out=%d speakers (n=%d, threshold=%.2f, Fa=%.3f, Fb=%.3f)",
            len(np.unique(ahc_clusters)),
            auto_num,
            len(np.unique(labels)),
            n,
            threshold,
            fa,
            fb,
        )

    return labels, centroids


def _centroids_from_labels(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean raw embedding per cluster id, indexed by label (0..max)."""
    k = int(labels.max()) + 1 if labels.size else 0
    return np.stack([embeddings[labels == c].mean(axis=0) for c in range(k)]).astype(np.float32)


def assign_speaker_labels(
    speaker_segments: list[tuple[float, float, int]],
    exclusive_segments: list[tuple[float, float, int]],
    centroids: np.ndarray,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]], np.ndarray, list[str]]:
    """Map integer cluster ids → ``SPEAKER_NN`` in sorted-id order.

    Mirrors upstream's ``classes()`` mapping zipped with sorted
    ``diarization.labels()``: the global ids present in the (overlap-aware)
    speaker timeline are sorted, the i-th becomes ``SPEAKER_{i:02d}``, and the
    returned embeddings are reordered to match. Returns
    ``(labeled_speaker, labeled_exclusive, ordered_embeddings, names)``.
    """
    present = sorted({cid for _, _, cid in speaker_segments})
    id_to_name = {cid: f"SPEAKER_{rank:02d}" for rank, cid in enumerate(present)}
    names = [id_to_name[cid] for cid in present]

    def _relabel(segs):
        return [(s, e, id_to_name[cid]) for s, e, cid in segs if cid in id_to_name]

    if centroids.shape[0] and present:
        ordered_embeddings = centroids[present]
    else:
        ordered_embeddings = np.zeros((0, centroids.shape[1] if centroids.ndim == 2 else 0), dtype=np.float32)
    return _relabel(speaker_segments), _relabel(exclusive_segments), ordered_embeddings, names


@dataclass
class SDResult:
    """Structured diarization result returned by :class:`PyannoteOnnxClient`."""

    speaker: list[DiarizationSegment]  # overlap-aware timeline
    exclusive: list[DiarizationSegment]  # non-overlapping timeline (for ASR)
    embeddings: np.ndarray  # (num_speakers, dim), in ``speaker_names`` order
    speaker_names: list[str]  # SPEAKER_00, SPEAKER_01, ...


def _build_reconstruction_inputs(
    *,
    chunks: list[ChunkSegmentation],
    metadata: list[EmbeddingMetadata],
    labels: np.ndarray,
    frame_duration: float,
    audio_duration: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], int, int]:
    """Build per-chunk soft / binary / label arrays for count-based reconstruction.

    For every chunk produces ``(frames, NUM_LOCAL_SPEAKERS)`` soft per-local
    activations (marginal probability) and 0/1 binarized activity, plus a
    ``(NUM_LOCAL_SPEAKERS,)`` map from local speaker → global cluster id
    (``-2`` when the local speaker was filtered out before clustering).
    Returns ``(soft, binary, labels_per_chunk, offsets_frames, total_frames,
    num_global)``.
    """
    # (chunk_idx, local_speaker_id) → global cluster label
    label_lookup: dict[tuple[int, int], int] = {}
    for meta, label in zip(metadata, labels.tolist(), strict=True):
        label_lookup[(meta.chunk_idx, meta.local_speaker_id)] = int(label)

    frames_per_window = chunks[0].probs.shape[0]
    padded_frames = round((chunks[-1].offset_sec + frames_per_window * frame_duration) / frame_duration)
    total_frames = min(padded_frames, round(audio_duration / frame_duration))
    num_global = int(labels.max()) + 1 if labels.size else 0

    soft_per_chunk: list[np.ndarray] = []
    binary_per_chunk: list[np.ndarray] = []
    labels_per_chunk: list[np.ndarray] = []
    offsets_frames: list[int] = []
    for chunk_idx, chunk in enumerate(chunks):
        per_speaker = _per_speaker_probability(chunk.probs)  # {1,2,3: (frames,)}
        soft = np.stack([per_speaker[s] for s in (1, 2, 3)], axis=1)
        binary = np.stack(
            [_hysteresis_binarize(per_speaker[s], onset=_ONSET, offset=_OFFSET) for s in (1, 2, 3)],
            axis=1,
        ).astype(np.float64)
        chunk_labels = np.array([label_lookup.get((chunk_idx, s), -2) for s in (1, 2, 3)], dtype=np.int64)
        soft_per_chunk.append(soft)
        binary_per_chunk.append(binary)
        labels_per_chunk.append(chunk_labels)
        offsets_frames.append(round(chunk.offset_sec / frame_duration))

    return soft_per_chunk, binary_per_chunk, labels_per_chunk, offsets_frames, total_frames, num_global


def _segments_to_diarization(labeled: list[tuple[float, float, str]], audio_duration: float) -> list[DiarizationSegment]:
    """Convert ``(start, end, speaker_name)`` tuples to sorted DiarizationSegments."""
    ordered = sorted(labeled, key=lambda s: (s[0], s[1]))
    return [
        DiarizationSegment(id=idx, start=float(start), end=float(min(end, audio_duration)), speaker=name)
        for idx, (start, end, name) in enumerate(ordered)
    ]


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

    def __call__(
        self,
        *,
        audio_input: np.ndarray,
        sample_rate: int,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        hook=None,
    ) -> SDResult:
        logger = logging.getLogger("pyannote_onnx_sd")

        def _emit(step: str, artifact=None):
            if hook is not None:
                hook(step, artifact)

        logger.debug(
            "audio shape=%s dtype=%s duration=%.1fs",
            audio_input.shape,
            audio_input.dtype,
            audio_input.size / sample_rate if audio_input.size else 0,
        )
        if audio_input.size == 0:
            return _empty_result()

        # 1. Per-chunk segmentation forward pass
        chunks, frame_duration = run_segmentation(
            audio=audio_input,
            sample_rate=sample_rate,
            session=self.seg_session,
            window_duration=_WINDOW_DURATION,
            window_step=_WINDOW_STEP,
        )
        if not chunks:
            return _empty_result()
        _emit("segmentation", chunks)

        # 2. Per-chunk per-local-speaker binarization + single-active mask
        speaker_masks = binarize_per_chunk(chunks, onset=_ONSET, offset=_OFFSET)
        if not speaker_masks:
            return _empty_result()

        # 3. Per-(chunk, local) embedding extraction with mask conditioning
        embeddings, metadata = extract_embeddings_per_chunk_speaker(
            sample_rate=sample_rate,
            session=self.emb_session,
            chunks=chunks,
            speaker_masks=speaker_masks,
            embedding_exclude_overlap=self.config.embedding_exclude_overlap,
        )
        if embeddings.shape[0] == 0:
            return _empty_result()
        _emit("embeddings", embeddings)

        # 4. PLDA + VBx clustering (community-1 algorithm) → labels + centroids
        plda = self._load_plda()
        labels, centroids = cluster_embeddings_vbx(
            embeddings,
            plda=plda,
            threshold=self.config.clustering_threshold,
            fa=self.config.vbx_fa,
            fb=self.config.vbx_fb,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
        )
        _emit("clustering", labels)

        # 5. Count-based reconstruction → overlap-aware + exclusive timelines
        audio_duration = audio_input.size / sample_rate
        soft, binary, labels_per_chunk, offsets, total_frames, num_global = _build_reconstruction_inputs(
            chunks=chunks,
            metadata=metadata,
            labels=labels,
            frame_duration=frame_duration,
            audio_duration=audio_duration,
        )
        speaker_segs, exclusive_segs = reconstruct_from_chunks(
            soft_per_chunk=soft,
            binary_per_chunk=binary,
            labels_per_chunk=labels_per_chunk,
            offsets_frames=offsets,
            total_frames=total_frames,
            num_global=num_global,
            frame_duration=frame_duration,
            min_duration_on=self.config.min_duration_on,
            min_duration_off=self.config.min_duration_off,
        )

        # 6. SPEAKER_NN labelling (sorted-id order) + embedding alignment
        labeled_sp, labeled_ex, ordered_emb, names = assign_speaker_labels(
            speaker_segs, exclusive_segs, centroids
        )
        result = SDResult(
            speaker=_segments_to_diarization(labeled_sp, audio_duration),
            exclusive=_segments_to_diarization(labeled_ex, audio_duration),
            embeddings=ordered_emb,
            speaker_names=names,
        )
        _emit("diarization", result.speaker)
        logger.info(
            "pyannote_onnx SD: %d chunks → %d (chunk,speaker) masks → %d embeddings → %d speakers → %d/%d segments",
            len(chunks),
            len(speaker_masks),
            embeddings.shape[0],
            len(names),
            len(result.speaker),
            len(result.exclusive),
        )
        return result


def _empty_result() -> SDResult:
    return SDResult(speaker=[], exclusive=[], embeddings=np.zeros((0, 256), dtype=np.float32), speaker_names=[])
