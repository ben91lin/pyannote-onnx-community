"""ABI surface for ONNX-based Pyannote Speaker Diarization.

Pure ABI layer: zero business logic. Two responsibilities:
1. ONNX session factories with path-keyed caches (so repeat loads share state).
2. Sliding-window iteration helper (pure NumPy, deterministic, zero-pads tail).

Mirrors the ``whisper_cpp_lib.py`` discipline — the
business logic (binarize, clustering, multi-stage refinement) lives in
a separate cache module.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from huggingface_hub import try_to_load_from_cache

try:
    import onnxruntime
except ImportError:  # pragma: no cover — gated by extras at install time
    onnxruntime = None  # type: ignore[assignment]


__all__ = [
    "hysteresis_binarize",
    "is_available",
    "iter_windows",
    "load_embedding_session",
    "load_segmentation_session",
    "resolve_onnx_path",
]


def hysteresis_binarize(prob: np.ndarray, onset: float, offset: float) -> np.ndarray:
    """Two-threshold binarization (mirrors pyannote.audio Binarize).

    When ``onset == offset`` this collapses to a simple threshold (strict
    ``>``/``<`` comparisons; equal-to-threshold inputs stay in the prior
    state). Used by both VAD and SD post-processing on per-frame probability
    arrays.
    """
    active = np.zeros_like(prob, dtype=bool)
    state = False
    for i, p in enumerate(prob):
        if state:
            if p < offset:
                state = False
        else:
            if p > onset:
                state = True
        active[i] = state
    return active


_seg_cache: dict[tuple[str, tuple[str, ...]], onnxruntime.InferenceSession] = {}
_emb_cache: dict[tuple[str, tuple[str, ...]], onnxruntime.InferenceSession] = {}


def is_available() -> bool:
    """Whether onnxruntime is importable in the current environment."""
    return onnxruntime is not None


def _new_session(model_path: str, providers=None) -> onnxruntime.InferenceSession:
    if onnxruntime is None:
        raise ImportError(
            "onnxruntime is required for pyannote_onnx SD provider. "
            "Install via `pip install pyannote-onnx-community`."
        )
    options = onnxruntime.SessionOptions()
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=options,
        providers=list(providers) if providers else ["CPUExecutionProvider"],
    )


def load_segmentation_session(model_path: str, providers=None) -> onnxruntime.InferenceSession:
    """Load (or return cached) segmentation ONNX session keyed by (path, providers)."""
    key = (model_path, tuple(providers) if providers else ("CPUExecutionProvider",))
    sess = _seg_cache.get(key)
    if sess is None:
        sess = _new_session(model_path, providers=providers)
        _seg_cache[key] = sess
    return sess


def load_embedding_session(model_path: str, providers=None) -> onnxruntime.InferenceSession:
    """Load (or return cached) embedding ONNX session keyed by (path, providers)."""
    key = (model_path, tuple(providers) if providers else ("CPUExecutionProvider",))
    sess = _emb_cache.get(key)
    if sess is None:
        sess = _new_session(model_path, providers=providers)
        _emb_cache[key] = sess
    return sess


def iter_windows(
    audio: np.ndarray,
    *,
    sample_rate: int,
    window_duration: float,
    window_step: float,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(offset_seconds, chunk)`` for sliding windows over ``audio``.

    Each chunk is exactly ``int(window_duration * sample_rate)`` samples; the
    final window is zero-padded on the right when audio runs short. Empty
    input yields nothing. Caller is responsible for passing mono float32 audio
    at ``sample_rate`` as mono float32.

    Footgun: because the final window is zero-padded, the padded frame grid can
    extend past the real audio. Consumers that build a timeline from these
    windows MUST clamp their output back to the true ``audio.size / sample_rate``
    duration, or they will report activity past the end of the input. See
    ``_assemble_global_timeline`` (SD) and ``_stitched_speech_probability``
    (VAD) for the two existing clamps.
    """
    if audio.size == 0:
        return
    window_samples = int(window_duration * sample_rate)
    step_samples = int(window_step * sample_rate)
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError(f"window/step must be > 0 (got window={window_duration}s, step={window_step}s)")

    start = 0
    while start < audio.size:
        end = start + window_samples
        chunk = audio[start:end]
        if chunk.size < window_samples:
            chunk = np.pad(chunk, (0, window_samples - chunk.size), mode="constant")
        yield start / sample_rate, chunk
        if end >= audio.size:
            break
        start += step_samples


def resolve_onnx_path(repo_or_path: str, filename: str = "onnx/model.onnx") -> str:
    """Resolve a HF repo id or absolute path to an on-disk ONNX file.

    Resolution order:

    1. If ``repo_or_path`` is an existing file, return it as-is.
    2. Otherwise treat it as a HF repo id and use
       ``huggingface_hub.try_to_load_from_cache`` to find the cached file
       (fast, offline-friendly): first ``filename``, then ``"model.onnx"``.
    3. If not cached, download via ``huggingface_hub.hf_hub_download`` —
       first ``filename``, then ``"model.onnx"`` — so a fresh user gets the
       model out-of-the-box.

    Args:
        repo_or_path: An absolute filesystem path to an ONNX file, or a
            HuggingFace repo id (e.g. ``"onnx-community/pyannote-segmentation-3.0"``).
        filename: Relative filename within the repo to look up.

    Returns:
        Absolute path to the ONNX file on disk.

    Raises:
        FileNotFoundError: When the file cannot be found locally, in the
            HuggingFace cache, or via download (e.g. offline, or the repo/file
            does not exist).
    """
    p = Path(repo_or_path)
    if p.is_file():
        return str(p)

    # 2. Cache fast path. ``try_to_load_from_cache`` may return a special
    # non-string sentinel (_CACHED_NO_EXIST) for a recorded-missing file, so
    # treat anything that is not a str as "not cached".
    cached = try_to_load_from_cache(repo_id=repo_or_path, filename=filename)
    if not isinstance(cached, str):
        # Fallback: file might be at the root, try without the onnx/ subdir.
        cached = try_to_load_from_cache(repo_id=repo_or_path, filename="model.onnx")
    if isinstance(cached, str):
        return cached

    # 3. Not cached — download. Import lazily so the cost only hits when needed.
    import huggingface_hub

    try:
        return huggingface_hub.hf_hub_download(repo_id=repo_or_path, filename=filename)
    except Exception as exc:
        try:
            return huggingface_hub.hf_hub_download(repo_id=repo_or_path, filename="model.onnx")
        except Exception as exc2:
            raise FileNotFoundError(
                f"Could not resolve {repo_or_path!r} to an ONNX file. "
                f"Tried filenames: {filename!r}, 'model.onnx' (cache + download). "
                f"You may be offline, or the repo/file does not exist."
            ) from exc2
