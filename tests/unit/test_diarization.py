import numpy as np
import pytest
from pyannote.core import Annotation

pytest.importorskip("kaldi_native_fbank")


class _FakeSeg:
    """Returns a 2-speaker segmentation: local spk 1 first half, spk 2 second."""

    def run(self, _outs, feed):
        x = feed["input_values"]  # (B, 1, samples)
        b = x.shape[0]
        frames = 50
        logits = np.full((b, frames, 7), -10.0, dtype=np.float32)
        logits[:, : frames // 2, 1] = 10.0  # local speaker A
        logits[:, frames // 2 :, 2] = 10.0  # local speaker B
        return [logits]


class _FakeEmb:
    """Two well-separated embeddings keyed by fbank energy mean sign."""

    def run(self, _outs, feed):
        feats = feed["input_features"]  # (1, T, 80)
        v = np.zeros((1, 256), dtype=np.float32)
        if float(feats.mean()) >= 0:
            v[0, 0] = 1.0
        else:
            v[0, 1] = 1.0
        return [v]


def _fake_cluster(embeddings, **_kw):
    """Stub PLDA+VBx: label by embedding argmax; return (labels, centroids)."""
    labels = embeddings.argmax(axis=1).astype(np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    k = int(labels.max()) + 1 if labels.size else 0
    centroids = np.stack([embeddings[labels == c].mean(axis=0) for c in range(k)]).astype(np.float32)
    return labels, centroids


def _make_dia(monkeypatch):
    from pyannote_onnx_community import diarization

    dia = diarization.ONNXSpeakerDiarization(seg_session=_FakeSeg(), emb_session=_FakeEmb())
    import pyannote_onnx_community._pipeline as pipe

    monkeypatch.setattr(pipe, "cluster_embeddings_vbx", _fake_cluster)
    return dia


def _audio():
    # normalised to [-1, 1] to satisfy load_audio's ndarray contract.
    return (np.random.default_rng(0).standard_normal(16000 * 12) * 0.1).astype(np.float32)


def test_call_returns_diarize_output(monkeypatch):
    from pyannote_onnx_community.output import DiarizeOutput

    dia = _make_dia(monkeypatch)
    out = dia(_audio())

    assert isinstance(out, DiarizeOutput)
    assert isinstance(out.speaker_diarization, Annotation)
    assert isinstance(out.exclusive_speaker_diarization, Annotation)


def test_speaker_labels_use_speaker_nn_format(monkeypatch):
    dia = _make_dia(monkeypatch)
    out = dia(_audio())
    assert len(out.speaker_diarization.labels()) >= 1
    assert all(lbl.startswith("SPEAKER_") for lbl in out.speaker_diarization.labels())


def test_speaker_embeddings_aligned_with_labels(monkeypatch):
    dia = _make_dia(monkeypatch)
    out = dia(_audio())
    assert out.speaker_embeddings is not None
    assert out.speaker_embeddings.shape[0] == len(out.speaker_diarization.labels())


def test_exclusive_timeline_has_no_overlap(monkeypatch):
    dia = _make_dia(monkeypatch)
    out = dia(_audio())
    # get_overlap() yields a Timeline of regions with >= 2 simultaneous speakers
    assert len(out.exclusive_speaker_diarization.get_overlap()) == 0


def test_hook_called_with_stage_names(monkeypatch):
    dia = _make_dia(monkeypatch)
    steps = []
    dia(_audio(), hook=lambda step, artifact=None: steps.append(step))
    assert "segmentation" in steps
    assert "diarization" in steps


def test_num_speakers_accepted_without_warning(monkeypatch):
    import warnings

    dia = _make_dia(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        out = dia(_audio(), num_speakers=2)
    from pyannote_onnx_community.output import DiarizeOutput

    assert isinstance(out, DiarizeOutput)


def test_num_speakers_accepted_positionally(monkeypatch):
    from pyannote_onnx_community.output import DiarizeOutput

    dia = _make_dia(monkeypatch)
    out = dia(_audio(), 2)  # positional num_speakers, mirrors upstream apply(file, 2)
    assert isinstance(out, DiarizeOutput)


def test_diarize_output_exported_from_package():
    import pyannote_onnx_community as pkg

    assert hasattr(pkg, "DiarizeOutput")
