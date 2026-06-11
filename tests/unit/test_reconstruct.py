import numpy as np

from pyannote_onnx_community._reconstruct import (
    aggregate_overlap_add,
    discrete_to_segments,
    reconstruct_from_chunks,
    top_count_discrete,
)


def _reconstruct(soft, binary, labels, num_global, *, offsets=None, total=None, frame=1.0):
    return reconstruct_from_chunks(
        soft_per_chunk=soft,
        binary_per_chunk=binary,
        labels_per_chunk=labels,
        offsets_frames=offsets if offsets is not None else [0] * len(soft),
        total_frames=total if total is not None else soft[0].shape[0],
        num_global=num_global,
        frame_duration=frame,
        min_duration_on=0.0,
        min_duration_off=0.0,
    )


def test_aggregate_sum_overlapping_chunks():
    c0 = np.ones((4, 1), dtype=np.float64)
    c1 = np.full((4, 1), 3.0)
    out = aggregate_overlap_add([c0, c1], [0, 2], total_frames=6, average=False)
    assert out[:, 0].tolist() == [1, 1, 4, 4, 3, 3]


def test_aggregate_average_divides_by_overlap_weight():
    c0 = np.ones((4, 1), dtype=np.float64)
    c1 = np.full((4, 1), 3.0)
    out = aggregate_overlap_add([c0, c1], [0, 2], total_frames=6, average=True)
    assert out[:, 0].tolist() == [1, 1, 2, 2, 3, 3]


def test_aggregate_uncovered_frames_stay_zero():
    c0 = np.ones((2, 1), dtype=np.float64)
    out = aggregate_overlap_add([c0], [0], total_frames=4, average=True)
    assert out[:, 0].tolist() == [1, 1, 0, 0]


def test_top_count_keeps_highest_per_frame():
    activations = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.7]])
    count = np.array([1, 2], dtype=np.uint8)
    out = top_count_discrete(activations, count)
    assert out[0].tolist() == [0, 1, 0]
    assert out[1].tolist() == [1, 0, 1]


def test_top_count_zero_speakers_is_silence():
    activations = np.array([[0.5, 0.4]])
    out = top_count_discrete(activations, np.array([0], dtype=np.uint8))
    assert out[0].tolist() == [0, 0]


def test_top_count_clamped_to_num_speakers():
    activations = np.array([[0.5, 0.4]])
    out = top_count_discrete(activations, np.array([5], dtype=np.uint8))
    assert out[0].tolist() == [1, 1]


def test_discrete_to_segments_basic_runs():
    # col0 active frames 0,1,2 ; col1 active frames 4,5 ; frame=0.5s
    discrete = np.zeros((6, 2))
    discrete[0:3, 0] = 1
    discrete[4:6, 1] = 1
    segs = discrete_to_segments(discrete, 0.5, min_duration_on=0.0, min_duration_off=0.0)
    assert (0.0, 1.5, 0) in segs
    assert (2.0, 3.0, 1) in segs


def test_discrete_to_segments_merges_short_off_gap():
    discrete = np.zeros((4, 1))
    discrete[[0, 1, 3], 0] = 1  # one-frame (0.5s) gap at frame 2
    segs = discrete_to_segments(discrete, 0.5, min_duration_on=0.0, min_duration_off=1.0)
    assert segs == [(0.0, 2.0, 0)]


def test_discrete_to_segments_drops_short_on_run():
    discrete = np.zeros((4, 1))
    discrete[1, 0] = 1  # single 0.5s run
    segs = discrete_to_segments(discrete, 0.5, min_duration_on=0.6, min_duration_off=0.0)
    assert segs == []


def test_reconstruct_sequential_speakers():
    soft = [np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])]
    binary = [np.array([[1, 0], [1, 0], [0, 1], [0, 1]])]
    labels = [np.array([0, 1])]
    speaker, exclusive = _reconstruct(soft, binary, labels, num_global=2)
    assert (0.0, 2.0, 0) in speaker
    assert (2.0, 4.0, 1) in speaker
    # No overlap here, so exclusive == speaker.
    assert sorted(exclusive) == sorted(speaker)


def test_reconstruct_overlap_exclusive_drops_quieter_speaker():
    soft = [np.array([[0.8, 0.7]])]
    binary = [np.array([[1, 1]])]  # 2 instantaneous speakers
    labels = [np.array([0, 1])]
    speaker, exclusive = _reconstruct(soft, binary, labels, num_global=2)
    # overlap-aware keeps both speakers in [0,1)
    assert (0.0, 1.0, 0) in speaker
    assert (0.0, 1.0, 1) in speaker
    # exclusive keeps only the louder one (col 0 @ 0.8)
    assert exclusive == [(0.0, 1.0, 0)]


def test_reconstruct_cluster_max_merges_local_speakers():
    # two local speakers mapped to the SAME global cluster -> max activation
    soft = [np.array([[0.3, 0.6]])]
    binary = [np.array([[0, 1]])]
    labels = [np.array([0, 0])]
    speaker, _ = _reconstruct(soft, binary, labels, num_global=1)
    assert speaker == [(0.0, 1.0, 0)]


def test_reconstruct_inactive_local_speaker_skipped():
    # local speaker 1 has label -2 (filtered) -> never painted
    soft = [np.array([[0.9, 0.9]])]
    binary = [np.array([[1, 0]])]
    labels = [np.array([0, -2])]
    speaker, _ = _reconstruct(soft, binary, labels, num_global=1)
    assert speaker == [(0.0, 1.0, 0)]
