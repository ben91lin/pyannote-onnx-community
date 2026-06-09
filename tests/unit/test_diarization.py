import numpy as np
import pytest
from pyannote.core import Annotation

pytest.importorskip("kaldi_native_fbank")


class _FakeSeg:
    """Returns a 2-speaker segmentation: local spk 1 first half, spk 2 second."""

    def run(self, _outs, feed):
        x = feed["input_values"]  # (B, 1, samples)
        b = x.shape[0]
        frames = 50
        logits = np.full((b, frames, 7), -10.0, dtype=np.float32)
        logits[:, : frames // 2, 1] = 10.0  # local speaker A
        logits[:, frames // 2 :, 2] = 10.0  # local speaker B
        return [logits]


class _FakeEmb:
    """Two well-separated embeddings keyed by fbank energy mean sign."""

    def run(self, _outs, feed):
        feats = feed["input_features"]  # (1, T, 80)
        v = np.zeros((1, 256), dtype=np.float32)
        if float(feats.mean()) >= 0:
            v[0, 0] = 1.0
        else:
            v[0, 1] = 1.0
        return [v]


def test_diarization_returns_annotation(monkeypatch):
    from pyannote_onnx_community import diarization

    dia = diarization.ONNXSpeakerDiarization(seg_session=_FakeSeg(), emb_session=_FakeEmb())

    # Bypass real PLDA+VBx: stub the clustering to label by embedding argmax.
    import pyannote_onnx_community._pipeline as pipe

    def _fake_cluster(embeddings, **_kw):
        return embeddings.argmax(axis=1).astype(np.int64)

    monkeypatch.setattr(pipe, "cluster_embeddings_vbx", _fake_cluster)

    audio = np.random.default_rng(0).standard_normal(16000 * 12).astype(np.float32)
    ann = dia(audio)
    assert isinstance(ann, Annotation)
    assert len(ann.labels()) >= 1
