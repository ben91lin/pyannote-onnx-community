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
