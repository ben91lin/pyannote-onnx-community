def test_load_session_threads_providers(monkeypatch):
    import pyannote_onnx_community._lib as lib

    captured = {}

    class _FakeSession:
        def __init__(self, model_path, sess_options=None, providers=None):
            captured["providers"] = providers

    monkeypatch.setattr(lib.onnxruntime, "InferenceSession", _FakeSession)
    lib.clear_session_cache()
    want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    lib.load_segmentation_session("/tmp/fake_seg.onnx", providers=want)
    assert captured["providers"] == want
    lib.load_embedding_session("/tmp/fake_emb.onnx", providers=want)
    assert captured["providers"] == want


def test_load_session_defaults_to_cpu(monkeypatch):
    import pyannote_onnx_community._lib as lib

    captured = {}

    class _FakeSession:
        def __init__(self, model_path, sess_options=None, providers=None):
            captured["providers"] = providers

    monkeypatch.setattr(lib.onnxruntime, "InferenceSession", _FakeSession)
    lib.clear_session_cache()
    lib.load_segmentation_session("/tmp/fake_seg2.onnx")
    assert captured["providers"] == ["CPUExecutionProvider"]
