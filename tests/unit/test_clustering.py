"""cluster_embeddings_vbx: centroid return + num/min/max force-count."""

import numpy as np
import pytest

from pyannote_onnx_community._pipeline import assign_speaker_labels, cluster_embeddings_vbx


class _MockPLDA:
    """Passthrough projection + identity Phi (mirrors test_pipeline mock)."""

    def __init__(self, lda_dim: int = 128):
        self.phi = np.ones(lda_dim, dtype=np.float32)
        self._lda_dim = lda_dim

    def __call__(self, emb: np.ndarray) -> np.ndarray:
        return emb[:, : self._lda_dim].astype(np.float64)


def _two_groups(dim=256, per=4):
    rng = np.random.RandomState(0)
    a = rng.randn(per, dim) * 0.01 + np.eye(1, dim, 0)
    b = rng.randn(per, dim) * 0.01 + np.eye(1, dim, 1)
    return np.vstack([a, b]).astype(np.float32)


def test_returns_labels_and_centroids():
    emb = _two_groups()
    labels, centroids = cluster_embeddings_vbx(emb, plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8)
    n = len(np.unique(labels))
    assert centroids.shape == (n, emb.shape[1])


def test_empty_returns_empty_labels_and_centroids():
    labels, centroids = cluster_embeddings_vbx(
        np.zeros((0, 256), dtype=np.float32), plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8
    )
    assert labels.size == 0
    assert centroids.shape[0] == 0


def test_single_returns_single():
    labels, centroids = cluster_embeddings_vbx(
        np.ones((1, 256), dtype=np.float32), plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8
    )
    assert labels.tolist() == [0]
    assert centroids.shape == (1, 256)


def test_num_clusters_forces_exact_count():
    emb = _two_groups(per=4)  # 8 embeddings, naturally ~2 groups
    labels, centroids = cluster_embeddings_vbx(
        emb, plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8, num_clusters=3
    )
    assert len(np.unique(labels)) == 3
    assert centroids.shape == (3, emb.shape[1])


def test_max_clusters_caps_count():
    emb = _two_groups(per=4)
    labels, _ = cluster_embeddings_vbx(
        emb, plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8, max_clusters=1
    )
    assert len(np.unique(labels)) == 1


# ---------------------------------------------------------------------------
# Invalid speaker-count arguments fail loud (before fast paths)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"num_clusters": 0}, {"num_clusters": -1}, {"min_clusters": 0}, {"max_clusters": -2}])
def test_nonpositive_bounds_raise(kwargs):
    emb = _two_groups(per=4)
    with pytest.raises(ValueError):
        cluster_embeddings_vbx(emb, plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8, **kwargs)


def test_min_greater_than_max_raises():
    emb = _two_groups(per=4)
    with pytest.raises(ValueError):
        cluster_embeddings_vbx(
            emb, plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8, min_clusters=3, max_clusters=2
        )


def test_invalid_bounds_raise_before_single_fast_path():
    # single embedding would hit the fast path; validation must run first.
    with pytest.raises(ValueError):
        cluster_embeddings_vbx(
            np.ones((1, 256), dtype=np.float32), plda=_MockPLDA(), threshold=0.7, fa=0.07, fb=0.8, num_clusters=0
        )


# ---------------------------------------------------------------------------
# assign_speaker_labels — SPEAKER_NN in sorted-id order + aligned embeddings
# ---------------------------------------------------------------------------


def test_assign_labels_sorted_order_and_embedding_alignment():
    # global ids 2 and 0 present; centroids indexed by global id (3 rows).
    speaker = [(0.0, 1.0, 2), (1.0, 2.0, 0)]
    exclusive = [(0.0, 1.0, 2)]
    centroids = np.array([[10.0], [11.0], [12.0]])  # id0->10, id1->11, id2->12

    labeled_sp, labeled_ex, ordered_emb, names = assign_speaker_labels(speaker, exclusive, centroids)

    # sorted present ids [0, 2] -> SPEAKER_00, SPEAKER_01
    assert names == ["SPEAKER_00", "SPEAKER_01"]
    # id0 -> SPEAKER_00, id2 -> SPEAKER_01
    assert (0.0, 1.0, "SPEAKER_01") in [(s, e, lbl) for s, e, lbl in labeled_sp]
    assert (1.0, 2.0, "SPEAKER_00") in [(s, e, lbl) for s, e, lbl in labeled_sp]
    assert labeled_ex == [(0.0, 1.0, "SPEAKER_01")]
    # embeddings ordered to match names: row0=id0 centroid(10), row1=id2 centroid(12)
    assert ordered_emb[:, 0].tolist() == [10.0, 12.0]
