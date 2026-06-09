import numpy as np
import pytest

pytest.importorskip("kaldi_native_fbank")


class _FakeEmb:
    def run(self, _outs, feed):
        feats = feed["input_features"]
        assert feats.ndim == 3 and feats.shape[2] == 80
        return [np.array([[3.0, 4.0] + [0.0] * 254], dtype=np.float32)]


def test_embedding_is_l2_normalised():
    from pyannote_onnx_community.embedding import ONNXSpeakerEmbedding

    emb = ONNXSpeakerEmbedding(emb_session=_FakeEmb())
    vec = emb(np.ones(16000 * 3, dtype=np.float32))
    assert vec.shape == (256,)
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-5)
    np.testing.assert_allclose(vec[:2], [0.6, 0.8], rtol=1e-5)
