import numpy as np


def test_imports():
    from pyannote_onnx_community import _lib, _plda, _vbx  # noqa: F401


def test_l2_norm_unit_norm():
    from pyannote_onnx_community._vbx import l2_norm

    rng = np.random.default_rng(0)
    fea = rng.standard_normal((20, 64)).astype(np.float32)
    normed = l2_norm(fea)
    np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(l2_norm(np.array([3.0, 4.0])), [0.6, 0.8], rtol=1e-6)


def test_iter_windows_zero_pads_tail():
    from pyannote_onnx_community._lib import iter_windows

    audio = np.ones(16000 * 7, dtype=np.float32)  # 7s
    windows = list(iter_windows(audio, sample_rate=16000, window_duration=5.0, window_step=0.5))
    assert windows, "expected at least one window"
    for _offset, chunk in windows:
        assert chunk.shape[0] == 16000 * 5  # every chunk padded to 5s
