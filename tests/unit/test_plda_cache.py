import pytest

from pyannote_onnx_community import _plda


@pytest.fixture(autouse=True)
def clean_cache():
    _plda._plda_cache.clear()
    yield
    _plda._plda_cache.clear()


def test_load_plda_loads_once(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(repo_id, subfolder="plda"):
        calls.append((repo_id, subfolder))
        return sentinel

    monkeypatch.setattr(_plda.PLDA, "from_pretrained", fake_from_pretrained)
    assert _plda.load_plda("repo") is sentinel
    assert _plda.load_plda("repo") is sentinel
    assert calls == [("repo", "plda")]


def test_failed_load_is_not_cached(monkeypatch):
    results = [None, "ok"]
    monkeypatch.setattr(_plda.PLDA, "from_pretrained", lambda repo_id, subfolder="plda": results.pop(0))
    assert _plda.load_plda("repo") is None
    assert _plda.load_plda("repo") == "ok"


def test_client_uses_cached_plda_and_explicit_plda_wins(monkeypatch):
    from pyannote_onnx_community._pipeline import PyannoteOnnxClient
    from pyannote_onnx_community.config import SDConfig

    shared = object()
    monkeypatch.setattr(_plda, "load_plda", lambda repo_id: shared)
    assert PyannoteOnnxClient(seg_session=None, emb_session=None, config=SDConfig())._load_plda() is shared
    explicit = object()
    client = PyannoteOnnxClient(seg_session=None, emb_session=None, config=SDConfig(), plda=explicit)
    assert client._load_plda() is explicit


def test_client_raises_when_plda_missing(monkeypatch):
    from pyannote_onnx_community._pipeline import PyannoteOnnxClient
    from pyannote_onnx_community.config import SDConfig

    monkeypatch.setattr(_plda, "load_plda", lambda repo_id: None)
    with pytest.raises(RuntimeError, match="PLDA"):
        PyannoteOnnxClient(seg_session=None, emb_session=None, config=SDConfig())._load_plda()
