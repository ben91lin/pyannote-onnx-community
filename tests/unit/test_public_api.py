import sys


def test_public_exports():
    import pyannote_onnx_community as p

    assert hasattr(p, "ONNXSpeakerDiarization")
    assert hasattr(p, "ONNXVoiceActivityDetection")
    assert hasattr(p, "ONNXSpeakerEmbedding")


def test_runtime_is_torch_free():
    # Importing the package (and its inference modules) must not pull torch.
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
    import pyannote_onnx_community  # noqa: F401
    from pyannote_onnx_community import diarization, embedding, vad  # noqa: F401

    assert "torch" not in sys.modules, "runtime path must not import torch"
