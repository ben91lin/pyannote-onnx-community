"""Generate per-stage golden arrays from upstream pyannote.audio PyTorch.

Run in the dev venv:
    .venv-dev/bin/python scripts/make_goldens.py

Writes tests/goldens/stage_goldens.npz with deterministic inputs + upstream
outputs for: l2_norm, vbx_setup, cluster_vbx, plda_projection, fbank,
segmentation. Stages whose upstream dependency is unavailable are skipped
(warning printed, key omitted; the parity test skips absent keys).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests" / "goldens"


def _synth_embeddings(n_speakers=3, n_per=15, dim=256, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_speakers, dim)) * 5.0
    rows = []
    for c in centers:
        rows.append(c + rng.standard_normal((n_per, dim)) * 0.5)
    return np.vstack(rows).astype(np.float32)


def _find_community1_snapshot():
    base = Path.home() / ".cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots"
    if not base.is_dir():
        return None
    for snap in base.iterdir():
        if (snap / "plda/plda.npz").is_file() and (snap / "plda/xvec_transform.npz").is_file():
            return snap
    return None


def _find_wespeaker_onnx():
    """Locate a cached WeSpeaker ONNX so we can instantiate the real upstream
    ``ONNXWeSpeakerPretrainedSpeakerEmbedding`` and call its bound
    ``compute_fbank`` method (the method itself is pure torchaudio kaldi fbank;
    the session is only needed to construct the object)."""
    hub = Path.home() / ".cache/huggingface/hub"
    for repo in (
        "models--onnx-community--wespeaker-voxceleb-resnet34-LM",
        "models--pyannote--wespeaker-voxceleb-resnet34-LM",
    ):
        snaps = hub / repo / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in snaps.iterdir():
            hits = list(snap.rglob("*.onnx"))
            if hits:
                return hits[0]
    return None


def main():
    out: dict[str, np.ndarray] = {}
    snapshot = _find_community1_snapshot()

    # --- l2_norm, vbx_setup, cluster_vbx, plda_projection ---
    try:
        from pyannote.audio.utils.vbx import (
            cluster_vbx as up_cluster_vbx,
            l2_norm as up_l2_norm,
            vbx_setup as up_vbx_setup,
        )

        fea_raw = _synth_embeddings()
        out["l2norm_in"] = fea_raw
        out["l2norm_up"] = np.asarray(up_l2_norm(fea_raw))

        if snapshot is not None:
            from pyannote_onnx_community._plda import PLDA

            xvec_npz = snapshot / "plda/xvec_transform.npz"
            plda_npz = snapshot / "plda/plda.npz"
            _xvec, _plda, psi_up = up_vbx_setup(xvec_npz, plda_npz)
            out["vbx_psi_up"] = np.asarray(psi_up)

            plda = PLDA(xvec_npz, plda_npz)
            fea = plda(fea_raw)
            out["plda_in"] = fea_raw
            out["plda_up"] = np.asarray(fea)

            from sklearn.cluster import AgglomerativeClustering

            ahc = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.5
            ).fit_predict(fea)
            np.random.seed(0)
            gamma_up, pi_up = up_cluster_vbx(ahc.copy(), fea, plda.phi, Fa=0.07, Fb=0.8, maxIters=20)
            out["vbx_ahc_in"] = ahc.astype(np.int64)
            out["vbx_fea_in"] = np.asarray(fea)
            out["vbx_phi_in"] = np.asarray(plda.phi)
            out["vbx_gamma_up"] = np.asarray(gamma_up)
            out["vbx_pi_up"] = np.asarray(pi_up)
        else:
            print("WARN: community-1 PLDA snapshot absent — skipping vbx/plda goldens")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: pyannote.audio vbx primitives unavailable: {e}")

    # --- fbank (pyannote compute_fbank) ---
    clip, sr = sf.read(GOLDENS / "clip.wav", dtype="float32")
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    chunk = clip[: sr * 3].astype(np.float32)
    out["fbank_in_audio"] = chunk
    out["fbank_in_sr"] = np.asarray([sr], dtype=np.int64)
    try:
        import torch
        from pyannote.audio.pipelines.speaker_verification import (
            ONNXWeSpeakerPretrainedSpeakerEmbedding as _Emb,
        )

        # The real upstream `compute_fbank` (speaker_verification.py:528-566) is an
        # INSTANCE method:
        #   def compute_fbank(self, waveforms, num_mel_bins=80, frame_length=25,
        #                     frame_shift=10, dither=0.0) -> Tensor
        #   waveforms : (batch_size, num_channels, num_samples)
        # It does `waveforms * (1 << 15)`, runs torchaudio.compliance.kaldi.fbank
        # per waveform (window_type="hamming", use_energy=False,
        # sample_frequency=self.sample_rate==16000), then subtracts the time-mean
        # (`features - features.mean(dim=1, keepdim=True)`). It needs only
        # `self.sample_rate`; the ONNX session is required solely to construct the
        # object, so we point it at a cached WeSpeaker ONNX and call the bound
        # method. This mirrors pyannote_onnx_community._pipeline._compute_fbank.
        feats_up = None
        onnx_path = _find_wespeaker_onnx()
        if onnx_path is not None:
            emb = _Emb(str(onnx_path), device=torch.device("cpu"))
            # (batch=1, channels=1, samples)
            wf = torch.from_numpy(chunk[np.newaxis, np.newaxis, :])
            with torch.no_grad():
                feats_up = emb.compute_fbank(wf)
        else:
            print("WARN: no cached WeSpeaker ONNX found — skipping fbank golden")
        if feats_up is not None:
            arr = feats_up.detach().cpu().numpy()
            out["fbank_up"] = np.squeeze(arr).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: upstream fbank unavailable: {e}")

    # --- segmentation probs (PyTorch pyannote/segmentation-3.0) ---
    try:
        import torch
        from pyannote.audio import Model

        seg = Model.from_pretrained("pyannote/segmentation-3.0")
        seg.eval()
        win = clip[: sr * 5].astype(np.float32)
        if win.shape[0] < sr * 5:
            win = np.pad(win, (0, sr * 5 - win.shape[0]))
        with torch.no_grad():
            logits = seg(torch.from_numpy(win[np.newaxis, np.newaxis, :]))
        arr = logits.detach().cpu().numpy()[0]
        e = np.exp(arr - arr.max(axis=-1, keepdims=True))
        probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        out["seg_in_window"] = win
        out["seg_up_probs"] = probs
    except Exception as e:  # noqa: BLE001
        print(f"WARN: upstream segmentation unavailable: {e}")

    GOLDENS.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDENS / "stage_goldens.npz", **out)
    print(f"wrote {GOLDENS / 'stage_goldens.npz'} with keys: {sorted(out)}")


if __name__ == "__main__":
    main()
