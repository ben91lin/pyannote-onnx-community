import numpy as np
from pyannote.core import Annotation


class _FakeSeg:
    """First 5s speech (class 1 active), rest silence (class 0)."""

    def run(self, _outs, feed):
        x = feed["input_values"]
        b, frames = x.shape[0], 50
        logits = np.full((b, frames, 7), -10.0, dtype=np.float32)
        logits[:, :, 1] = 10.0  # speaker A active across the whole window
        return [logits]


def test_vad_returns_annotation_with_speech():
    from pyannote_onnx_community.vad import ONNXVoiceActivityDetection

    vad = ONNXVoiceActivityDetection(seg_session=_FakeSeg())
    audio = np.ones(16000 * 8, dtype=np.float32)
    ann = vad(audio)
    assert isinstance(ann, Annotation)
    total = sum(seg.duration for seg in ann.get_timeline())
    assert total > 0


def test_vad_short_audio_does_not_exceed_duration():
    """P1: 1s audio is zero-padded to a full 5s window; VAD output must not
    report speech past the real 1s input (relaxed durations so the short clip
    survives filtering — the point under test is the upper bound)."""
    from pyannote_onnx_community.config import VADConfig
    from pyannote_onnx_community.vad import ONNXVoiceActivityDetection

    cfg = VADConfig(min_duration_on=0.1, min_duration_off=0.1)
    vad = ONNXVoiceActivityDetection(seg_session=_FakeSeg(), config=cfg)  # 50 frames/window → 0.1s
    audio = np.ones(16000 * 1, dtype=np.float32)  # 1s
    ann = vad(audio)
    assert len(ann) >= 1
    assert max(seg.end for seg in ann.get_timeline()) <= 1.0 + 0.1


def test_segments_from_active_mask_filters_short():
    from pyannote_onnx_community.vad import _segments_from_active_mask

    active = np.array([True, True, False, True], dtype=bool)  # 0.2s, 0.1s runs
    out = _segments_from_active_mask(active, 0.1, min_duration_on=0.15, min_duration_off=0.0)
    assert out == [(0.0, 0.2)]
