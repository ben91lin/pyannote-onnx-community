import pytest


def test_local_file_passthrough(tmp_path):
    from pyannote_onnx_community._lib import resolve_onnx_path

    f = tmp_path / "model.onnx"
    f.write_bytes(b"\x00")
    assert resolve_onnx_path(str(f)) == str(f)


def test_cache_hit_returns_cached(monkeypatch):
    import pyannote_onnx_community._lib as lib

    monkeypatch.setattr(lib, "try_to_load_from_cache", lambda repo_id, filename: "/cached/model.onnx")
    # hf_hub_download must NOT be called on a cache hit:
    import huggingface_hub

    def _boom(*a, **k):
        raise AssertionError("hf_hub_download should not be called on cache hit")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    assert lib.resolve_onnx_path("some/repo") == "/cached/model.onnx"


def test_downloads_when_not_cached(monkeypatch):
    import pyannote_onnx_community._lib as lib
    import huggingface_hub

    monkeypatch.setattr(lib, "try_to_load_from_cache", lambda repo_id, filename: None)
    calls = {}

    def _fake_dl(repo_id, filename):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return f"/downloaded/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_dl)
    out = lib.resolve_onnx_path("onnx-community/pyannote-segmentation-3.0")
    assert out == "/downloaded/onnx/model.onnx"
    assert calls["repo_id"] == "onnx-community/pyannote-segmentation-3.0"


def test_raises_filenotfound_when_download_fails(monkeypatch):
    import pyannote_onnx_community._lib as lib
    import huggingface_hub

    monkeypatch.setattr(lib, "try_to_load_from_cache", lambda repo_id, filename: None)

    def _fail(repo_id, filename):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail)
    with pytest.raises(FileNotFoundError):
        lib.resolve_onnx_path("nonexistent/repo")
