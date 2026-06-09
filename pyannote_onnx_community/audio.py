"""Audio loading: decode any container to mono float32 at a target sample rate.

`load_audio` accepts a path/str, a file-like object, or an np.ndarray (already
sampled at `sample_rate`; downmixed to mono and cast to float32). Path/file
input is decoded + resampled via PyAV (no torch).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def _downmix(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    # (channels, samples) or (samples, channels) -> average the channel axis.
    axis = 0 if wav.shape[0] < wav.shape[-1] else -1
    return wav.mean(axis=axis)


def load_audio(audio, sample_rate: int = 16000) -> np.ndarray:
    """Return mono float32 PCM at `sample_rate`."""
    if isinstance(audio, np.ndarray):
        return _downmix(audio).astype(np.float32).reshape(-1)

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
