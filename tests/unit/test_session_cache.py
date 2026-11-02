import pytest

import pyannote_onnx_community._lib as lib
from pyannote_onnx_community.config import SessionConfig


class _FakeSession:
    created = []

    def __init__(self, model_path, sess_options=None, providers=None):
        self.options = sess_options
        self.providers = providers
        _FakeSession.created.append(self)


@pytest.fixture(autouse=True)
def fake_ort(monkeypatch):
    _FakeSession.created = []
    monkeypatch.setattr(lib.onnxruntime, "InferenceSession", _FakeSession)
    lib.clear_session_cache()
    yield
    lib.clear_session_cache()


def test_equal_config_shares_session():
    a = lib.get_session("m", SessionConfig(intra_op_threads=2))
    b = lib.get_session("m", SessionConfig(intra_op_threads=2))
    assert a is b
    assert len(_FakeSession.created) == 1


def test_different_threads_or_providers_do_not_share():
    a = lib.get_session("m", SessionConfig(intra_op_threads=1))
    b = lib.get_session("m", SessionConfig(intra_op_threads=2))
    c = lib.get_session("m", SessionConfig(providers=["P"]))
    assert len({id(a), id(b), id(c)}) == 3


def test_threads_reach_session_options():
    s = lib.get_session("m", SessionConfig(intra_op_threads=3))
    assert s.options.intra_op_num_threads == 3


def test_provider_options_reach_session():
    opts = {"device_id": 1}
    s = lib.get_session("m", SessionConfig(providers=[("P", opts)]))
    assert s.providers == [("P", opts)]


def test_legacy_loaders_accept_provider_options():
    s = lib.load_segmentation_session("m", providers=[("P", {"device_id": 1})])
    assert s.providers == [("P", {"device_id": 1})]


def test_legacy_loaders_share_with_get_session():
    assert lib.load_embedding_session("m") is lib.get_session("m")


def test_clear_session_cache():
    a = lib.get_session("m")
    lib.clear_session_cache()
    assert lib.get_session("m") is not a


def test_handle_with_injected_session_never_resolves(monkeypatch):
    monkeypatch.setattr(lib, "resolve_onnx_path", lambda *a: pytest.fail("resolved"))
    injected = object()
    handle = lib.make_session_handle("repo", session=injected, session_config=None, providers=None)
    assert handle.get() is injected


def test_handle_rejects_session_and_config():
    with pytest.raises(ValueError):
        lib.make_session_handle("repo", session=object(), session_config=SessionConfig(), providers=None)


def test_handle_resolves_now_but_builds_later(monkeypatch):
    resolved = []
    monkeypatch.setattr(lib, "resolve_onnx_path", lambda *a: resolved.append(a) or "m")
    handle = lib.make_session_handle("repo", session=None, session_config=None, providers=["P"])
    assert resolved and _FakeSession.created == []
    assert handle.get().providers == ["P"]


def test_handle_config_beats_providers(monkeypatch):
    monkeypatch.setattr(lib, "resolve_onnx_path", lambda *a: "m")
    handle = lib.make_session_handle(
        "repo", session=None, session_config=SessionConfig(providers=["C"]), providers=["P"]
    )
    assert handle.get().providers == ["C"]


def test_after_fork_hook_leaks_and_clears(monkeypatch):
    incref = []
    monkeypatch.setattr(lib, "_leak_reference", incref.append)
    s = lib.get_session("m")
    old_lock = lib._session_lock
    lib._after_fork_in_child()
    assert incref == [s]
    assert lib._session_cache == {}
    assert lib._session_lock is not old_lock
