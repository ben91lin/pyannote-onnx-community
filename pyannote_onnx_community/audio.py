"""Audio loading: decode any container to mono float32 at a target sample rate.

`load_audio` accepts a path/str, a file-like object, or an np.ndarray. Path/file
input is decoded + resampled via PyAV (no torch) and normalised to float32.

ndarray input is trusted, not converted (matching whisper's contract): it must
already be a floating-point waveform, normalised to [-1, 1], at the target
`sample_rate` (mono 1-D, or 2-D ``(channels, samples)`` which is downmixed). It is NOT resampled or
PCM-normalised — pass a path/file object if you need decoding. Integer PCM or
clearly un-normalised arrays raise ``ValueError`` rather than producing
plausible-but-wrong output.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def _downmix(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        # 2-D is (channels, samples) -> average the channel axis.
        return wav.mean(axis=0)
    raise ValueError(
        f"ndarray audio must be 1-D (mono) or 2-D (channels, samples), got {wav.ndim}-D."
    )


def load_audio(audio, sample_rate: int = 16000) -> np.ndarray:
    """Return mono float32 PCM at `sample_rate`.

    For ndarray input ``sample_rate`` is not applied (no resampling); the array
    is trusted to already match it. See the module docstring for the contract.
    """
    if isinstance(audio, np.ndarray):
        if not np.issubdtype(audio.dtype, np.floating):
            raise ValueError(
                f"ndarray audio must be a floating-point waveform, got dtype {audio.dtype}. "
                "Integer PCM is not auto-normalised; convert first, e.g. "
                "`arr.astype(np.float32) / 32768.0`, or pass a file path/object to decode."
            )
        wav = _downmix(audio).astype(np.float32).reshape(-1)
        if wav.size:
            if not np.isfinite(wav).all():
                raise ValueError("ndarray audio contains NaN/Inf; expected finite samples in [-1, 1].")
            if np.abs(wav).max() > 1.0 + 1e-3:
                raise ValueError(
                    "ndarray audio is not normalised to [-1, 1] (max abs value exceeds 1); "
                    "looks like un-normalised PCM. Divide by the PCM full-scale (e.g. 32768.0), "
                    "or pass a file path/object to decode."
                )
        return wav

    import av

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    raw = io.BytesIO()
    src = str(audio) if isinstance(audio, (str, Path)) else audio
    with av.open(src, mode="r", metadata_errors="ignore") as container:
        for frame in container.decode(audio=0):
            frame.pts = None
            for resampled in resampler.resample(frame):
                raw.write(resampled.to_ndarray().tobytes())
        for resampled in resampler.resample(None):  # flush
            raw.write(resampled.to_ndarray().tobytes())

    pcm = np.frombuffer(raw.getbuffer(), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm
