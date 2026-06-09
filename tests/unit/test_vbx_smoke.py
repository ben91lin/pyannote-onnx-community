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

    # 6.3s (100 800 samples): the last window (offset 1.5s) ends at 6.5s,
    # which is past the audio end, so np.pad must supply 0.3s of zeros.
    audio = np.ones(16000 * 6 + 4800, dtype=np.float32)
    windows = list(iter_windows(audio, sample_rate=16000, window_duration=5.0, window_step=0.5))
    assert windows, "expected at least one window"
    for _offset, chunk in windows:
        assert chunk.shape[0] == 16000 * 5  # every chunk padded to 5s

    # The final window must contain a genuinely zero-padded tail.
    last_offset, last_chunk = windows[-1]
    audio_duration = audio.size / 16000  # 6.3
    pad_start = int(round((audio_duration - last_offset) * 16000))
    assert pad_start < last_chunk.shape[0], "expected the last window to extend past the audio"
    assert np.all(last_chunk[pad_start:] == 0.0), "tail beyond audio end must be zero-padded"
