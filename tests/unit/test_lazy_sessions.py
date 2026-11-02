import pytest

pytest.importorskip("kaldi_native_fbank")

import pyannote_onnx_community._lib as lib  # noqa: E402
from pyannote_onnx_community import (  # noqa: E402
    ONNXSpeakerDiarization,
    ONNXSpeakerEmbedding,
    ONNXVoiceActivityDetection,
    SessionConfig,
)


class _FakeSession:
    created = []

    def __init__(self, model_path, sess_options=None, providers=None):
        self.model_path = model_path
        self.options = sess_options
        self.providers = providers
        _FakeSession.created.append(self)


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    _FakeSession.created = []
    plda_loads = []
    monkeypatch.setattr(lib.onnxruntime, "InferenceSession", _FakeSession)
    monkeypatch.setattr(lib, "resolve_onnx_path", lambda repo, filename="onnx/model.onnx": f"/fake/{repo}")
    monkeypatch.setattr(
        "pyannote_onnx_community._plda.load_plda", lambda repo_id: plda_loads.append(repo_id) or object()
    )
    lib.clear_session_cache()
    yield plda_loads
    lib.clear_session_cache()


@pytest.mark.parametrize("cls", [ONNXSpeakerDiarization, ONNXVoiceActivityDetection, ONNXSpeakerEmbedding])
def test_constructor_creates_no_session(cls):
    cls()
    assert _FakeSession.created == []


def test_diarization_preload_builds_once_and_loads_plda(fake_runtime):
    dia = ONNXSpeakerDiarization()
    dia.preload()
    assert len(_FakeSession.created) == 2
    assert fake_runtime == [dia.config.plda_repo_id]
    dia.preload()
    assert len(_FakeSession.created) == 2


@pytest.mark.parametrize("cls", [ONNXVoiceActivityDetection, ONNXSpeakerEmbedding])
def test_single_model_preload_builds_once(cls):
    obj = cls()
    obj.preload()
    obj.preload()
    assert len(_FakeSession.created) == 1


def test_vad_and_diarization_share_segmentation_session():
    cfg = SessionConfig(intra_op_threads=2)
    dia = ONNXSpeakerDiarization(seg_session_config=cfg)
    vad = ONNXVoiceActivityDetection(seg_session_config=SessionConfig(intra_op_threads=2))
    assert dia._seg is vad._seg
    assert len(_FakeSession.created) == 1


def test_different_thread_configs_do_not_share():
    dia = ONNXSpeakerDiarization(seg_session_config=SessionConfig(intra_op_threads=1))
    vad = ONNXVoiceActivityDetection(seg_session_config=SessionConfig(intra_op_threads=2))
    assert dia._seg is not vad._seg


def test_providers_and_config_reach_sessions():
    dia = ONNXSpeakerDiarization(
        providers=["P"], seg_session_config=SessionConfig(providers=["C"], intra_op_threads=3)
    )
    assert dia._seg.providers == ["C"]
    assert dia._seg.options.intra_op_num_threads == 3
    assert dia._emb.providers == ["P"]


def test_provider_options_accepted():
    emb = ONNXSpeakerEmbedding(providers=[("CUDAExecutionProvider", {"device_id": 1})])
    assert emb._emb.providers == [("CUDAExecutionProvider", {"device_id": 1})]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seg_session": object(), "seg_session_config": SessionConfig()},
        {"emb_session": object(), "emb_session_config": SessionConfig()},
    ],
)
def test_injected_session_plus_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ONNXSpeakerDiarization(**kwargs)


def test_injected_session_returned_as_is():
    seg = object()
    dia = ONNXSpeakerDiarization(seg_session=seg)
    assert dia._seg is seg
    assert len(_FakeSession.created) == 0


def test_seg_property_returns_live_session():
    dia = ONNXSpeakerDiarization()
    assert dia._seg is lib.get_session("/fake/onnx-community/pyannote-segmentation-3.0", SessionConfig())


def test_cleared_cache_is_rebuilt_through_instance():
    dia = ONNXSpeakerDiarization()
    first = dia._seg
    lib.clear_session_cache()
    assert dia._seg is not first
