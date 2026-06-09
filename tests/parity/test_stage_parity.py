"""Per-stage array parity: our torch-free ONNX impl vs upstream pyannote.audio.

Consumes committed goldens (tests/goldens/stage_goldens.npz) produced by
scripts/make_goldens.py from the official PyTorch pipeline. Runs WITHOUT torch.
Each stage skips if its golden key is absent.
"""

# Observed parity (vs upstream pyannote.audio 4.0.4 PyTorch, Task 10 goldens):
#   l2_norm: max abs diff 0.0
#   cluster_vbx gamma/pi: max abs diff 0.0
#   plda projection: max abs diff 0.0
#   fbank: cosine 1.0000001 (max abs diff ~2.5e-4)
#   segmentation: cosine 1.0000000 (max abs diff ~2.4e-7)

from pathlib import Path

import numpy as np
import pytest

GOLDENS = Path(__file__).resolve().parents[1] / "goldens" / "stage_goldens.npz"


@pytest.fixture(scope="module")
def g():
    if not GOLDENS.is_file():
        pytest.skip("goldens not generated — run scripts/make_goldens.py")
    return np.load(GOLDENS)


def _require(g, *keys):
    for k in keys:
        if k not in g.files:
            pytest.skip(f"golden key {k!r} absent")


def test_l2_norm_parity(g):
    _require(g, "l2norm_in", "l2norm_up")
    from pyannote_onnx_community._vbx import l2_norm

    np.testing.assert_allclose(l2_norm(g["l2norm_in"]), g["l2norm_up"], rtol=1e-5)


def test_cluster_vbx_parity(g):
    _require(g, "vbx_ahc_in", "vbx_fea_in", "vbx_phi_in", "vbx_gamma_up", "vbx_pi_up")
    from pyannote_onnx_community._vbx import cluster_vbx

    np.random.seed(0)
    gamma, pi = cluster_vbx(
        g["vbx_ahc_in"].copy(), g["vbx_fea_in"], g["vbx_phi_in"], Fa=0.07, Fb=0.8, maxIters=20
    )
    np.testing.assert_allclose(gamma, g["vbx_gamma_up"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(pi, g["vbx_pi_up"], rtol=1e-5, atol=1e-7)


def test_plda_projection_parity(g):
    _require(g, "plda_in", "plda_up")
    base = Path.home() / ".cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots"
    snap = next((s for s in base.iterdir() if (s / "plda/plda.npz").is_file()), None) if base.is_dir() else None
    if snap is None:
        pytest.skip("community-1 snapshot absent")
    from pyannote_onnx_community._plda import PLDA

    plda = PLDA(snap / "plda/xvec_transform.npz", snap / "plda/plda.npz")
    np.testing.assert_allclose(plda(g["plda_in"]), g["plda_up"], rtol=1e-5, atol=1e-7)


def test_fbank_parity(g):
    _require(g, "fbank_in_audio", "fbank_in_sr", "fbank_up")
    from pyannote_onnx_community._pipeline import _compute_fbank

    ours = _compute_fbank(g["fbank_in_audio"], int(g["fbank_in_sr"][0]))
    up = g["fbank_up"]
    n = min(ours.shape[0], up.shape[0])
    cos = float(
        np.sum(ours[:n] * up[:n]) / (np.linalg.norm(ours[:n]) * np.linalg.norm(up[:n]) + 1e-12)
    )
    assert cos >= 0.9999, f"fbank cosine {cos:.6f} below 0.9999"


def test_segmentation_parity(g):
    _require(g, "seg_in_window", "seg_up_probs")
    from pyannote_onnx_community._lib import load_segmentation_session, resolve_onnx_path

    try:
        sess = load_segmentation_session(
            resolve_onnx_path("onnx-community/pyannote-segmentation-3.0", "onnx/model.onnx")
        )
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"segmentation ONNX model unavailable (no cache / no network): {exc}")
    win = g["seg_in_window"].astype(np.float32)[np.newaxis, np.newaxis, :]
    logits = sess.run(None, {"input_values": win})[0][0]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    up = g["seg_up_probs"]
    n = min(probs.shape[0], up.shape[0])
    cos = float(np.sum(probs[:n] * up[:n]) / (np.linalg.norm(probs[:n]) * np.linalg.norm(up[:n]) + 1e-12))
    assert cos >= 0.999, f"segmentation cosine {cos:.6f} below 0.999"
