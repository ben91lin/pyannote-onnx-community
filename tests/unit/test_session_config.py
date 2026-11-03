import pytest

from pyannote_onnx_community.config import SessionConfig


def test_defaults():
    c = SessionConfig()
    assert c.intra_op_threads == 0
    assert c.ort_providers() == ["CPUExecutionProvider"]


def test_provider_options_are_hashable_and_round_trip():
    opts = {"device_id": 1, "arena_extend_strategy": "kSameAsRequested"}
    c = SessionConfig(providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])
    assert hash(c) == hash(SessionConfig(providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"]))
    assert c.ort_providers() == [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]


def test_unhashable_option_values_are_frozen_to_str():
    c = SessionConfig(providers=[("P", {"profiles": [1, 2], "extra": {"a": 1}})])
    hash(c)
    assert c.ort_providers() == [("P", {"extra": "{'a': 1}", "profiles": "[1, 2]"})]


def test_option_order_does_not_change_identity():
    a = SessionConfig(providers=[("P", {"a": 1, "b": 2})])
    b = SessionConfig(providers=[("P", {"b": 2, "a": 1})])
    assert a == b


def test_freezing_is_idempotent():
    a = SessionConfig(providers=[("P", {"a": 1})])
    assert SessionConfig(providers=a.providers) == a


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_invalid_threads(bad):
    with pytest.raises(ValueError):
        SessionConfig(intra_op_threads=bad)


def test_empty_providers_rejected():
    with pytest.raises(ValueError):
        SessionConfig(providers=())


def test_single_string_providers_rejected():
    with pytest.raises(ValueError):
        SessionConfig(providers="CPUExecutionProvider")


def test_from_env(monkeypatch):
    monkeypatch.setenv("X_INTRA_OP_THREADS", "4")
    assert SessionConfig.from_env("X").intra_op_threads == 4
    monkeypatch.setenv("X_INTRA_OP_THREADS", " ")
    assert SessionConfig.from_env("X").intra_op_threads == 0
    monkeypatch.delenv("X_INTRA_OP_THREADS")
    assert SessionConfig.from_env("X", providers=["P"]).ort_providers() == ["P"]


def test_from_env_invalid(monkeypatch):
    monkeypatch.setenv("X_INTRA_OP_THREADS", "many")
    with pytest.raises(ValueError, match="X_INTRA_OP_THREADS"):
        SessionConfig.from_env("X")
