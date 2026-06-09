"""End-to-end DER parity: our ONNX pipeline vs upstream pyannote.audio PyTorch.

Gated: requires the dev extra (pyannote.audio + torch + pyannote.metrics) and a
resolvable HF token for the community-1 pipeline. Skips cleanly otherwise.
Run with the dev venv:
    .venv-dev/bin/python -m pytest tests/e2e -v -s
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyannote.audio")
pytest.importorskip("pyannote.metrics")
pytest.importorskip("torch")
pytest.importorskip("soundfile")

CLIP = Path(__file__).resolve().parents[1] / "goldens" / "clip.wav"


def _resolve_token():
    import os

    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    # Modern huggingface_hub (>=0.19) exposes get_token(); HfFolder was removed
    # in 1.x. Try the modern API first, then fall back for older hubs.
    try:
        from huggingface_hub import get_token

        tok = get_token()
        if tok:
            return tok
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def test_der_drift_and_speaker_count():
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    if not CLIP.is_file():
        pytest.skip("clip.wav not present")
    token = _resolve_token()
    if token is None:
        pytest.skip("no resolvable HF token for the gated community-1 pipeline")

    wav, sr = sf.read(CLIP, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    up_pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    if up_pipe is None:
        pytest.skip("could not load upstream community-1 pipeline (token/access?)")
    up_out = up_pipe({"waveform": torch.from_numpy(wav[np.newaxis, :]), "sample_rate": sr})
    # pyannote.audio 4.x returns a DiarizeOutput; the Annotation is .speaker_diarization.
    # Older versions return a bare Annotation.
    ref = getattr(up_out, "speaker_diarization", up_out)

    from pyannote_onnx_community import ONNXSpeakerDiarization

    ours = ONNXSpeakerDiarization()(wav)

    der = DiarizationErrorRate()(ref, ours)
    print(f"\nDER(ours vs upstream)={der:.4f}  ours_speakers={len(ours.labels())}  ref_speakers={len(ref.labels())}")
    assert der < 0.20, f"DER {der:.3f} too high vs upstream"
    assert abs(len(ours.labels()) - len(ref.labels())) <= 1, (
        f"speaker count drift: ours={len(ours.labels())} ref={len(ref.labels())}"
    )
