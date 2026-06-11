import numpy as np
import pytest


def test_passthrough_ndarray_mono_float32():
    from pyannote_onnx_community.audio import load_audio

    wav = np.ones(16000, dtype=np.float64)
    out = load_audio(wav, sample_rate=16000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] == 16000


def test_ndarray_stereo_is_downmixed():
    from pyannote_onnx_community.audio import load_audio

    stereo = np.ones((2, 8000), dtype=np.float32)
    out = load_audio(stereo, sample_rate=16000)
    assert out.ndim == 1
    assert out.shape[0] == 8000


def test_ndarray_multichannel_downmixed_along_channel_axis():
    from pyannote_onnx_community.audio import load_audio

    # (channels, samples): two channels averaged into one sample each.
    wav = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)  # shape (2, 2)
    out = load_audio(wav, sample_rate=16000)
    assert out.ndim == 1
    assert out.shape[0] == 2
    np.testing.assert_allclose(out, [0.4, 0.6], rtol=1e-6)


def test_ndarray_single_channel_2d_keeps_samples():
    from pyannote_onnx_community.audio import load_audio

    # (1, N): one channel, N samples -> N mono samples.
    wav = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    out = load_audio(wav, sample_rate=16000)
    assert out.ndim == 1
    assert out.shape[0] == 3


def test_ndarray_two_channel_single_sample_downmixed():
    from pyannote_onnx_community.audio import load_audio

    # (2, 1): two channels, one sample -> single averaged mono sample.
    wav = np.array([[0.2], [0.6]], dtype=np.float32)
    out = load_audio(wav, sample_rate=16000)
    assert out.ndim == 1
    assert out.shape[0] == 1
    np.testing.assert_allclose(out, [0.4], rtol=1e-6)


def test_ndarray_three_dimensional_raises():
    from pyannote_onnx_community.audio import load_audio

    with pytest.raises(ValueError, match="1-D|2-D"):
        load_audio(np.ones((2, 2, 2), dtype=np.float32), sample_rate=16000)


def test_integer_pcm_ndarray_raises():
    from pyannote_onnx_community.audio import load_audio

    with pytest.raises(ValueError, match="floating-point"):
        load_audio(np.ones(16000, dtype=np.int16), sample_rate=16000)


def test_unnormalised_float_ndarray_raises():
    from pyannote_onnx_community.audio import load_audio

    with pytest.raises(ValueError, match="normalised"):
        load_audio(np.full(16000, 2.0, dtype=np.float32), sample_rate=16000)


def test_nonfinite_ndarray_raises():
    from pyannote_onnx_community.audio import load_audio

    with pytest.raises(ValueError, match="NaN/Inf"):
        load_audio(np.full(16000, np.nan, dtype=np.float32), sample_rate=16000)
