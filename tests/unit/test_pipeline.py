"""Unit tests for the ONNX community-1 SD pipeline.

Phase 2.5 restructure: no flat-timeline stitching; embedding extraction is
mask-conditioned per (chunk, local-speaker); multi-stage refinement removed.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from pyannote_onnx_community._pipeline import (
    ChunkSegmentation,
    ChunkSpeakerMask,
    EmbeddingMetadata,
    assemble_output,
    binarize_per_chunk,
    cluster_embeddings_vbx,
    extract_embeddings_per_chunk_speaker,
    run_segmentation,
)

# ---------------------------------------------------------------------------
# run_segmentation
# ---------------------------------------------------------------------------


def _mock_seg_session(constant_logits: np.ndarray) -> MagicMock:
    """Return a session whose .run(...) yields ``constant_logits`` every call.

    Handles both legacy single-window callers (B=1 input) and batched callers
    (B>1) by tiling the constant logits along the batch axis to match the
    input batch size.
    """
    sess = MagicMock()
    # constant_logits has shape (1, frames, classes) by convention here.

    def _run(_outputs, ort_inputs):
        batch = np.asarray(ort_inputs["input_values"])
        batch_size = batch.shape[0]
        # Tile constant_logits to match the requested batch.
        if constant_logits.shape[0] == batch_size:
            return [constant_logits]
        return [np.broadcast_to(constant_logits, (batch_size,) + constant_logits.shape[1:]).copy()]

    sess.run = MagicMock(side_effect=_run)
    sess.get_inputs = MagicMock(return_value=[MagicMock(name="input_values")])
    sess.get_inputs.return_value[0].name = "input_values"
    return sess


def test_run_segmentation_returns_one_entry_per_window():
    """5s window/5s step over 10s audio: 2 windows → 2 ChunkSegmentation entries."""
    fake_logits = np.zeros((1, 294, 7), dtype=np.float32)
    fake_logits[0, :, 1] = 8.0  # strong class-1 logit → softmax ~0.998
    sess = _mock_seg_session(fake_logits)
    audio = np.zeros(16000 * 10, dtype=np.float32)

    chunks, frame_duration = run_segmentation(
        audio=audio,
        sample_rate=16000,
        session=sess,
        window_duration=5.0,
        window_step=5.0,
    )

    # 2 windows are processed (one per ChunkSegmentation entry); with batching
    # they fit in a single session.run call (default batch_size=32).
    assert sess.run.call_count == 1
    assert len(chunks) == 2
    assert all(isinstance(c, ChunkSegmentation) for c in chunks)
    assert chunks[0].offset_sec == 0.0
    assert chunks[1].offset_sec == 5.0
    # frame_duration = window_duration / frames_per_window
    assert abs(frame_duration - 5.0 / 294) < 1e-9
    # Per-chunk softmax: probabilities sum to 1 per frame; class 1 dominates.
    for chunk in chunks:
        assert chunk.probs.shape == (294, 7)
        assert np.allclose(chunk.probs.sum(axis=-1), 1.0, atol=1e-5)
        assert (chunk.probs[:, 1] > 0.99).all()
        assert chunk.audio.shape == (16000 * 5,)


def test_run_segmentation_does_not_stitch_overlap():
    """2s window/1s step: overlapping windows produce independent chunks (no averaging)."""
    fake_logits = np.zeros((1, 100, 7), dtype=np.float32)
    fake_logits[0, :, 1] = 8.0
    sess = _mock_seg_session(fake_logits)
    audio = np.zeros(16000 * 4, dtype=np.float32)  # 4s → windows at 0,1,2

    chunks, frame_duration = run_segmentation(
        audio=audio,
        sample_rate=16000,
        session=sess,
        window_duration=2.0,
        window_step=1.0,
    )

    # 3 windows fire (offsets 0,1,2 — iter_windows stops once end >= audio.size)
    assert len(chunks) == 3
    assert [round(c.offset_sec, 2) for c in chunks] == [0.0, 1.0, 2.0]
    assert abs(frame_duration - 2.0 / 100) < 1e-9


def test_run_segmentation_audio_shorter_than_window_pads():
    fake_logits = np.zeros((1, 100, 7), dtype=np.float32)
    sess = _mock_seg_session(fake_logits)
    audio = np.zeros(16000 * 1, dtype=np.float32)  # 1s, window=5s

    chunks, frame_duration = run_segmentation(
        audio=audio,
        sample_rate=16000,
        session=sess,
        window_duration=5.0,
        window_step=5.0,
    )

    assert sess.run.call_count == 1
    assert len(chunks) == 1
    assert chunks[0].audio.shape == (16000 * 5,)  # zero-padded
    assert abs(frame_duration - 5.0 / 100) < 1e-9


def test_run_segmentation_empty_audio_returns_empty():
    sess = _mock_seg_session(np.zeros((1, 100, 7), dtype=np.float32))
    audio = np.array([], dtype=np.float32)

    chunks, frame_duration = run_segmentation(
        audio=audio,
        sample_rate=16000,
        session=sess,
        window_duration=5.0,
        window_step=5.0,
    )

    assert chunks == []
    assert frame_duration == 0.0
    assert sess.run.call_count == 0


def test_run_segmentation_passes_3d_input_with_channel_dim():
    """pyannote/segmentation-3.0 expects (batch, channels, samples) — verify channel dim added."""
    fake_logits = np.zeros((1, 100, 7), dtype=np.float32)
    sess = MagicMock()
    captured = []

    def _run(_outputs, ort_inputs):
        captured.append(ort_inputs["input_values"].shape)
        return [fake_logits]

    sess.run = MagicMock(side_effect=_run)
    audio = np.zeros(16000 * 5, dtype=np.float32)

    run_segmentation(
        audio=audio,
        sample_rate=16000,
        session=sess,
        window_duration=5.0,
        window_step=5.0,
    )
    assert captured[0] == (1, 1, 16000 * 5)


# ---------------------------------------------------------------------------
# _per_speaker_probability (powerset -> per-speaker marginal)
# ---------------------------------------------------------------------------


def test_per_speaker_probability_sums_membership_classes():
    """Powerset classes are mutually exclusive, so a speaker's marginal is the
    SUM of the classes it belongs to (matching VAD/upstream), not the max.

    Split mass: A=0.3 (class 1) and A+B=0.3 (class 4). Speaker A's marginal is
    0.6 (> 0.5 threshold); the old max-based code reported only 0.3 and would
    have dropped the speaker — a driver of the documented speaker under-count.
    """
    from pyannote_onnx_community._pipeline import _per_speaker_probability

    probs = np.zeros((1, 7), dtype=np.float32)
    probs[0, 1] = 0.3  # A
    probs[0, 4] = 0.3  # A+B
    out = _per_speaker_probability(probs)

    # Speaker 1 (A): classes 1, 4, 5 → 0.3 + 0.3 + 0.0 = 0.6
    np.testing.assert_allclose(out[1][0], 0.6, rtol=1e-6)
    # Speaker 2 (B): classes 2, 4, 6 → 0.0 + 0.3 + 0.0 = 0.3
    np.testing.assert_allclose(out[2][0], 0.3, rtol=1e-6)
    # Speaker 3 (C): classes 3, 5, 6 → 0.0
    np.testing.assert_allclose(out[3][0], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# binarize_per_chunk
# ---------------------------------------------------------------------------


def _chunk_with_class_active(class_idx: int, num_frames: int = 100, prob_high: float = 0.9) -> ChunkSegmentation:
    probs = np.full((num_frames, 7), 0.05, dtype=np.float32)
    probs[:, class_idx] = prob_high
    return ChunkSegmentation(offset_sec=0.0, audio=np.zeros(16000 * 5, dtype=np.float32), probs=probs)


def test_binarize_per_chunk_emits_one_mask_for_active_speaker():
    chunks = [_chunk_with_class_active(class_idx=1)]
    masks = binarize_per_chunk(chunks, onset=0.5, offset=0.5)
    assert len(masks) == 1
    assert masks[0].chunk_idx == 0
    assert masks[0].local_speaker_id == 1
    assert masks[0].binary_mask.all()
    # Single speaker active → single_active_mask True everywhere
    assert masks[0].single_active_mask.all()
    assert masks[0].is_overlap_dominant is False


def test_binarize_per_chunk_skips_speakers_that_never_fire():
    chunks = [_chunk_with_class_active(class_idx=1)]
    masks = binarize_per_chunk(chunks, onset=0.5, offset=0.5)
    speakers = {m.local_speaker_id for m in masks}
    assert speakers == {1}  # speakers 2 and 3 never fired


def test_binarize_per_chunk_overlap_class_marks_both_speakers():
    """Class 4 = speakers 1+2 simultaneously → both speakers see activity, overlap-dominant."""
    chunks = [_chunk_with_class_active(class_idx=4)]
    masks = binarize_per_chunk(chunks, onset=0.5, offset=0.5)
    speakers = {m.local_speaker_id for m in masks}
    assert speakers == {1, 2}
    # Both speakers active in same frames → single_active_mask all False
    for m in masks:
        assert not m.single_active_mask.any()
        assert m.is_overlap_dominant is True


def test_binarize_per_chunk_independent_chunks():
    """Each chunk binarized independently; same local id may map to different chunks."""
    chunks = [_chunk_with_class_active(class_idx=1), _chunk_with_class_active(class_idx=2)]
    chunks[1] = ChunkSegmentation(offset_sec=5.0, audio=chunks[1].audio, probs=chunks[1].probs)
    masks = binarize_per_chunk(chunks, onset=0.5, offset=0.5)
    by_chunk = {(m.chunk_idx, m.local_speaker_id) for m in masks}
    assert by_chunk == {(0, 1), (1, 2)}


def test_binarize_per_chunk_empty_chunks_returns_empty():
    assert binarize_per_chunk([], onset=0.5, offset=0.5) == []


def test_binarize_per_chunk_silent_chunk_emits_no_mask():
    """Non-speech chunk (class 0 dominant) → no speaker masks."""
    silent_probs = np.full((100, 7), 0.01, dtype=np.float32)
    silent_probs[:, 0] = 0.9
    chunks = [ChunkSegmentation(offset_sec=0.0, audio=np.zeros(16000 * 5, dtype=np.float32), probs=silent_probs)]
    masks = binarize_per_chunk(chunks, onset=0.5, offset=0.5)
    assert masks == []


# ---------------------------------------------------------------------------
# extract_embeddings_per_chunk_speaker
# ---------------------------------------------------------------------------


def _mock_emb_session(embedding_size: int = 256) -> MagicMock:
    sess = MagicMock()
    counter = {"calls": 0}

    def _run(_outputs, ort_inputs):
        counter["calls"] += 1
        emb = np.full((1, embedding_size), float(counter["calls"]), dtype=np.float32)
        return [emb]

    sess.run = MagicMock(side_effect=_run)
    return sess


def _all_active_chunk_with_audio(num_frames: int = 100, audio_seconds: float = 5.0) -> ChunkSegmentation:
    """Real audio chunk (silence) — kaldi_native_fbank still produces frames from silence."""
    return ChunkSegmentation(
        offset_sec=0.0,
        audio=np.zeros(int(16000 * audio_seconds), dtype=np.float32),
        probs=np.zeros((num_frames, 7), dtype=np.float32),
    )


def test_extract_embeddings_returns_one_per_chunk_speaker():
    """One chunk + two speakers fully active → 2 embeddings."""
    pytest.importorskip("kaldi_native_fbank")
    sess = _mock_emb_session()
    chunk = _all_active_chunk_with_audio(num_frames=100)
    full_active = np.ones(100, dtype=bool)
    masks = [
        ChunkSpeakerMask(0, 0.0, 1, full_active.copy(), full_active.copy(), False),
        ChunkSpeakerMask(0, 0.0, 2, full_active.copy(), full_active.copy(), False),
    ]
    embeddings, metadata = extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess,
        chunks=[chunk],
        speaker_masks=masks,
        embedding_exclude_overlap=False,
    )
    assert embeddings.shape == (2, 256)
    assert sess.run.call_count == 2
    assert [m.local_speaker_id for m in metadata] == [1, 2]


def test_extract_embeddings_skips_inactive_below_min_active_ratio():
    """Speaker active for < 20% of chunk frames → skipped."""
    pytest.importorskip("kaldi_native_fbank")
    sess = _mock_emb_session()
    chunk = _all_active_chunk_with_audio(num_frames=100)
    sparse = np.zeros(100, dtype=bool)
    sparse[:10] = True  # 10% active < 20%
    full_active = np.ones(100, dtype=bool)
    masks = [
        ChunkSpeakerMask(0, 0.0, 1, sparse, full_active, False),
        ChunkSpeakerMask(0, 0.0, 2, full_active, full_active, False),
    ]
    embeddings, metadata = extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess,
        chunks=[chunk],
        speaker_masks=masks,
        embedding_exclude_overlap=False,
    )
    assert embeddings.shape == (1, 256)
    assert metadata[0].local_speaker_id == 2


def test_extract_embeddings_caches_fbank_per_chunk():
    """Multiple speakers in one chunk → fbank computed once, session.run called per speaker."""
    pytest.importorskip("kaldi_native_fbank")
    from pyannote_onnx_community import _pipeline as cache_mod

    sess = _mock_emb_session()
    chunk = _all_active_chunk_with_audio(num_frames=100)
    full_active = np.ones(100, dtype=bool)
    masks = [
        ChunkSpeakerMask(0, 0.0, 1, full_active.copy(), full_active.copy(), False),
        ChunkSpeakerMask(0, 0.0, 2, full_active.copy(), full_active.copy(), False),
        ChunkSpeakerMask(0, 0.0, 3, full_active.copy(), full_active.copy(), False),
    ]
    fbank_calls = {"n": 0}
    real_fbank = cache_mod._compute_fbank

    def _wrapped_fbank(audio, sr):
        fbank_calls["n"] += 1
        return real_fbank(audio, sr)

    cache_mod._compute_fbank = _wrapped_fbank
    try:
        embeddings, _ = extract_embeddings_per_chunk_speaker(
            sample_rate=16000,
            session=sess,
            chunks=[chunk],
            speaker_masks=masks,
            embedding_exclude_overlap=False,
        )
    finally:
        cache_mod._compute_fbank = real_fbank
    assert embeddings.shape == (3, 256)
    assert fbank_calls["n"] == 1  # cached across speakers
    assert sess.run.call_count == 3


def test_extract_embeddings_l2_normalises():
    pytest.importorskip("kaldi_native_fbank")
    sess = _mock_emb_session()
    chunk = _all_active_chunk_with_audio(num_frames=100)
    full_active = np.ones(100, dtype=bool)
    masks = [ChunkSpeakerMask(0, 0.0, 1, full_active, full_active, False)]
    embeddings, _ = extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess,
        chunks=[chunk],
        speaker_masks=masks,
        embedding_exclude_overlap=False,
    )
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_extract_embeddings_empty_returns_empty():
    sess = MagicMock()
    embeddings, metadata = extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess,
        chunks=[],
        speaker_masks=[],
        embedding_exclude_overlap=False,
    )
    assert embeddings.shape == (0, 256)
    assert metadata == []
    sess.run.assert_not_called()


def test_extract_embeddings_exclude_overlap_uses_single_active_mask():
    """When embedding_exclude_overlap=True, frames where multiple speakers fire are masked out."""
    pytest.importorskip("kaldi_native_fbank")
    sess = _mock_emb_session()
    chunk = _all_active_chunk_with_audio(num_frames=100)
    # Speaker 1 active everywhere; only first half is single-active (rest is overlap)
    binary = np.ones(100, dtype=bool)
    single = np.zeros(100, dtype=bool)
    single[:50] = True  # 50% single-active
    masks = [ChunkSpeakerMask(0, 0.0, 1, binary, single, False)]
    embeddings, metadata = extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess,
        chunks=[chunk],
        speaker_masks=masks,
        embedding_exclude_overlap=True,
    )
    # 50% > 20% min_active_ratio → kept
    assert embeddings.shape == (1, 256)
    assert metadata[0].chunk_idx == 0


def test_extract_embeddings_skips_when_masked_fbank_too_short():
    """Mask leaves so few fbank frames that wespeaker would emit NaN → skip."""
    pytest.importorskip("kaldi_native_fbank")
    sess = _mock_emb_session()
    chunk = _all_active_chunk_with_audio(num_frames=100, audio_seconds=5.0)
    # Speaker active in only the very first segmentation frame; resampled mask
    # leaves ~5 fbank frames out of ~498 → below the 25-frame wespeaker threshold.
    binary = np.zeros(100, dtype=bool)
    binary[:21] = True  # 21% active — passes ratio filter
    single = np.ones(100, dtype=bool)
    masks = [ChunkSpeakerMask(0, 0.0, 1, binary, single, False)]
    embeddings, _ = extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess,
        chunks=[chunk],
        speaker_masks=masks,
        embedding_exclude_overlap=False,
        min_active_ratio=0.2,
    )
    # 21% of 5s ≈ 105 fbank frames after resample, > 25 — should be kept.
    assert embeddings.shape == (1, 256)
    # Now make the active region truly tiny (1 frame ≈ 5 fbank frames after resample)
    binary2 = np.zeros(100, dtype=bool)
    binary2[:20] = True  # exactly 20% — passes ratio (>= 0.2 * 100 = 20)
    masks2 = [ChunkSpeakerMask(0, 0.0, 1, binary2, single, False)]
    sess2 = _mock_emb_session()
    embeddings2, _ = extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess2,
        chunks=[chunk],
        speaker_masks=masks2,
        embedding_exclude_overlap=False,
        min_active_ratio=0.2,
    )
    # 20% of 498 ≈ 99 fbank frames, comfortably above 25 → kept.
    assert embeddings2.shape == (1, 256)


def test_extract_embeddings_passes_fbank_features_to_session():
    """Verify session receives input_features shape (1, T, 80) — masked Kaldi fbank."""
    pytest.importorskip("kaldi_native_fbank")

    sess = MagicMock()
    captured = []

    def _run(_outputs, ort_inputs):
        captured.append((dict(ort_inputs), ort_inputs[next(iter(ort_inputs))].shape))
        return [np.full((1, 256), 1.0, dtype=np.float32)]

    sess.run = MagicMock(side_effect=_run)
    chunk = _all_active_chunk_with_audio(num_frames=100, audio_seconds=5.0)
    full_active = np.ones(100, dtype=bool)
    masks = [ChunkSpeakerMask(0, 0.0, 1, full_active, full_active, False)]

    extract_embeddings_per_chunk_speaker(
        sample_rate=16000,
        session=sess,
        chunks=[chunk],
        speaker_masks=masks,
        embedding_exclude_overlap=False,
    )

    assert sess.run.call_count == 1
    inputs, shape = captured[0]
    assert "input_features" in inputs
    assert shape[0] == 1  # batch
    assert shape[2] == 80  # mel bins
    assert shape[1] >= 25  # at least wespeaker minimum


# ---------------------------------------------------------------------------
# cluster_embeddings_vbx (smoke — full parity covered by test_pyannote_vbx_parity.py)
# ---------------------------------------------------------------------------


class _MockPLDA:
    """Mock PLDA — passthrough projection + identity Phi."""

    def __init__(self, lda_dim: int = 128):
        self.phi = np.ones(lda_dim, dtype=np.float32)
        self._lda_dim = lda_dim

    def __call__(self, emb: np.ndarray) -> np.ndarray:
        return emb[:, : self._lda_dim].astype(np.float64)


def test_cluster_embeddings_vbx_empty_returns_empty():
    labels = cluster_embeddings_vbx(
        np.zeros((0, 256), dtype=np.float32),
        plda=_MockPLDA(),
        threshold=0.6,
        fa=0.07,
        fb=0.8,
    )
    assert labels.size == 0


def test_cluster_embeddings_vbx_single_returns_single():
    labels = cluster_embeddings_vbx(
        np.ones((1, 256), dtype=np.float32),
        plda=_MockPLDA(),
        threshold=0.6,
        fa=0.07,
        fb=0.8,
    )
    assert labels.tolist() == [0]


# ---------------------------------------------------------------------------
# assemble_output (unchanged behavior)
# ---------------------------------------------------------------------------


from pyannote_onnx_community._pipeline import DiarizationSegment  # noqa: E402


def test_assemble_output_maps_cluster_to_speaker_label():
    out = assemble_output([(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0)])
    assert len(out) == 3
    assert all(isinstance(s, DiarizationSegment) for s in out)
    speaker_labels = {s.speaker for s in out}
    assert len(speaker_labels) == 2
    # First and third (cluster 0) share a speaker
    assert out[0].speaker == out[2].speaker


def test_assemble_output_merges_same_cluster_close_in_time():
    out = assemble_output([(0.0, 1.0, 0), (1.2, 2.0, 0)])
    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == 2.0


def test_assemble_output_keeps_separate_when_gap_above_threshold():
    out = assemble_output([(0.0, 1.0, 0), (1.6, 2.0, 0)])
    assert len(out) == 2


def test_assemble_output_empty_list_returns_empty():
    assert assemble_output([]) == []


def test_assemble_output_assigns_increasing_id():
    out = assemble_output([(0.0, 1.0, 0), (2.0, 3.0, 1)])
    assert out[0].id == 0
    assert out[1].id == 1


# ---------------------------------------------------------------------------
# PyannoteOnnxClient end-to-end (sessions mocked)
# ---------------------------------------------------------------------------


from pyannote_onnx_community._pipeline import PyannoteOnnxClient  # noqa: E402
from pyannote_onnx_community.config import SDConfig  # noqa: E402


def _seg_session_one_speaker(num_frames_per_window: int = 100) -> MagicMock:
    """Mock session emitting raw LOGITS (run_segmentation applies softmax).

    Tiles output along the batch axis to match the input batch size, so
    the same fixture works whether the caller batches or not.
    """
    sess = MagicMock()
    template = np.full((1, num_frames_per_window, 7), -10.0, dtype=np.float32)
    template[0, :, 1] = 8.0  # strong speaker-1 logit → softmax ~0.998

    def _run(_outputs, ort_inputs):
        batch_size = np.asarray(ort_inputs["input_values"]).shape[0]
        return [np.broadcast_to(template, (batch_size, num_frames_per_window, 7)).copy()]

    sess.run = MagicMock(side_effect=_run)
    return sess


def test_client_single_speaker_yields_segments():
    """Audio entirely speaker 1 → at least one DiarizationSegment."""
    pytest.importorskip("kaldi_native_fbank")
    seg = _seg_session_one_speaker()
    emb = MagicMock()
    emb.run = MagicMock(return_value=[np.full((1, 256), 1.0, dtype=np.float32)])

    cfg = SDConfig()
    client = PyannoteOnnxClient(seg_session=seg, emb_session=emb, config=cfg, plda=_MockPLDA())
    audio = np.zeros(16000 * 10, dtype=np.float32)

    result = client(audio_input=audio, sample_rate=16000)

    assert len(result) >= 1
    assert all(s.speaker for s in result)


def test_client_empty_audio_returns_empty_list():
    seg = MagicMock()
    emb = MagicMock()
    cfg = SDConfig()
    client = PyannoteOnnxClient(seg_session=seg, emb_session=emb, config=cfg)
    audio = np.array([], dtype=np.float32)
    assert client(audio_input=audio, sample_rate=16000) == []
    seg.run.assert_not_called()
    emb.run.assert_not_called()


def test_client_does_not_emit_segments_past_audio_end():
    """P1: short audio (1s) is zero-padded to a full 5s window; the output
    timeline must clamp back to the real 1s input, not run to 5s."""
    pytest.importorskip("kaldi_native_fbank")
    seg = _seg_session_one_speaker()  # 100 frames/window → frame_duration 0.05s
    emb = MagicMock()
    emb.run = MagicMock(return_value=[np.full((1, 256), 1.0, dtype=np.float32)])

    cfg = SDConfig()
    client = PyannoteOnnxClient(seg_session=seg, emb_session=emb, config=cfg, plda=_MockPLDA())
    audio = np.zeros(16000 * 1, dtype=np.float32)  # 1s, padded to 5s window

    result = client(audio_input=audio, sample_rate=16000)

    assert len(result) >= 1
    frame_duration = 5.0 / 100
    assert max(s.end for s in result) <= 1.0 + frame_duration


def test_client_no_speech_logits_returns_empty_list():
    """All frames non-speech (class 0 wins after softmax) → no segments."""
    seg = MagicMock()
    silent_template = np.full((1, 100, 7), -10.0, dtype=np.float32)
    silent_template[0, :, 0] = 8.0  # strong non-speech logit

    def _silent_run(_outputs, ort_inputs):
        b = np.asarray(ort_inputs["input_values"]).shape[0]
        return [np.broadcast_to(silent_template, (b, 100, 7)).copy()]

    seg.run = MagicMock(side_effect=_silent_run)
    emb = MagicMock()

    cfg = SDConfig()
    client = PyannoteOnnxClient(seg_session=seg, emb_session=emb, config=cfg)
    audio = np.zeros(16000 * 5, dtype=np.float32)
    assert client(audio_input=audio, sample_rate=16000) == []
    emb.run.assert_not_called()


def test_client_runs_only_single_stage():
    """Phase 2.5: multi-stage refinement removed — each window fires segmentation once."""
    pytest.importorskip("kaldi_native_fbank")
    seg = _seg_session_one_speaker()
    emb = MagicMock()
    emb.run = MagicMock(return_value=[np.full((1, 256), 1.0, dtype=np.float32)])

    cfg = SDConfig()
    client = PyannoteOnnxClient(seg_session=seg, emb_session=emb, config=cfg, plda=_MockPLDA())
    audio = np.zeros(16000 * 35, dtype=np.float32)  # 35s

    client(audio_input=audio, sample_rate=16000)
    # 35s @ window_duration=5s, window_step=0.5s (matches upstream segmentation_step=0.1):
    # iter_windows emits chunks at offsets 0.0, 0.5, ..., 30.0 → 61 chunks. With
    # batched segmentation forward (default batch_size=32), 61 chunks → ceil(61/32)=2
    # session.run calls.
    assert seg.run.call_count == 2


# ---------------------------------------------------------------------------
# EmbeddingMetadata smoke (dataclass exists and is importable)
# ---------------------------------------------------------------------------


def test_embedding_metadata_is_a_dataclass_with_expected_fields():
    meta = EmbeddingMetadata(chunk_idx=2, chunk_offset_sec=10.0, local_speaker_id=1, frame_mask=np.zeros(5, dtype=bool))
    assert meta.chunk_idx == 2
    assert meta.chunk_offset_sec == 10.0
    assert meta.local_speaker_id == 1
    assert meta.frame_mask.shape == (5,)
