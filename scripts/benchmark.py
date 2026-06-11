"""Reproducible median speed benchmark for this ONNX pipeline vs PyTorch.

The benchmark matrix is:
    ours ONNX {CPU, CoreML} x pytorch {CPU, MPS}

Each input is decoded once via the package PyAV loader and then run through every
backend on the same waveform. Backends get one short warmup run before timed
full-audio runs. RTFx = audio seconds / wall-clock seconds (higher is faster).
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os

# Must be set before torch imports so MPS-unsupported ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402

SR = 16000
WARMUP_SEC = 10.0
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "benchmark_results.md"


@dataclass(frozen=True)
class InputClip:
    name: str
    path: Path
    private: bool
    audio: np.ndarray


@dataclass(frozen=True)
class InputSummary:
    name: str
    path: str
    private: bool
    audio_s: float


@dataclass(frozen=True)
class BackendConfig:
    engine: str
    device: str
    builder: Callable[[], tuple[Callable[[np.ndarray], int], str]]


@dataclass(frozen=True)
class BuiltBackend:
    engine: str
    device: str
    status: str
    run: Callable[[np.ndarray], int] | None
    providers: str
    reason: str


@dataclass(frozen=True)
class BenchmarkResult:
    clip: str
    path: str
    private: bool
    audio_s: float
    engine: str
    device: str
    status: str
    walls: list[float]
    median_wall: float
    median_rtfx: float
    speakers: int
    providers: str
    reason: str


@dataclass(frozen=True)
class EnvironmentInfo:
    benchmark_date: str
    platform: str
    machine: str
    chip: str
    python: str
    onnxruntime: str
    onnxruntime_providers: str
    torch: str
    torch_mps: str
    pyannote_audio: str


class BackendUnavailable(RuntimeError):
    """Raised by backend builders when a configured backend is unavailable."""


def _nan() -> float:
    return float("nan")


def _format_float(value: float, digits: int = 1) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _md_cell(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", r"\|")


def _providers_label(providers: list[str] | tuple[str, ...]) -> str:
    return ", ".join(provider.replace("ExecutionProvider", "") for provider in providers)


def _ours_runner(providers: list[str]) -> tuple[Callable[[np.ndarray], int], str]:
    """Return a runner and the actual ONNX Runtime providers used."""

    from pyannote_onnx_community import ONNXSpeakerDiarization

    dia = ONNXSpeakerDiarization(providers=providers)
    try:
        actual = _providers_label(dia._seg.get_providers())
    except Exception:  # noqa: BLE001
        actual = _providers_label(providers)

    def run(wav: np.ndarray) -> int:
        return len(dia(wav).speaker_diarization.labels())

    return run, actual


def _coreml_ours_runner() -> tuple[Callable[[np.ndarray], int], str]:
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(f"onnxruntime unavailable: {exc}") from exc

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise BackendUnavailable("CoreMLExecutionProvider unavailable")
    return _ours_runner(["CoreMLExecutionProvider", "CPUExecutionProvider"])


def _pytorch_runner(device: str) -> tuple[Callable[[np.ndarray], int], str]:
    """Return a runner for the upstream community-1 pyannote.audio pipeline."""

    import torch
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=get_token())
    pipe.to(torch.device(device))

    def run(wav: np.ndarray) -> int:
        waveform = torch.from_numpy(wav[np.newaxis, :])
        out = pipe({"waveform": waveform, "sample_rate": SR})
        ann = getattr(out, "speaker_diarization", out)
        return len(ann.labels())

    return run, device


def _pytorch_mps_runner() -> tuple[Callable[[np.ndarray], int], str]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(f"torch unavailable: {exc}") from exc

    if not torch.backends.mps.is_available():
        raise BackendUnavailable("torch MPS unavailable")
    return _pytorch_runner("mps")


def _build_backend_configs() -> list[BackendConfig]:
    return [
        BackendConfig("ours", "cpu", lambda: _ours_runner(["CPUExecutionProvider"])),
        BackendConfig("ours", "coreml", _coreml_ours_runner),
        BackendConfig("pytorch", "cpu", lambda: _pytorch_runner("cpu")),
        BackendConfig("pytorch", "mps", _pytorch_mps_runner),
    ]


def _build_backend(backend: BackendConfig) -> BuiltBackend:
    try:
        run, providers = backend.builder()
    except BackendUnavailable as exc:
        return BuiltBackend(
            engine=backend.engine,
            device=backend.device,
            status="skip",
            run=None,
            providers="",
            reason=str(exc) or repr(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return BuiltBackend(
            engine=backend.engine,
            device=backend.device,
            status="fail",
            run=None,
            providers="",
            reason=f"build failed: {exc}",
        )

    return BuiltBackend(
        engine=backend.engine,
        device=backend.device,
        status="ok",
        run=run,
        providers=providers,
        reason="",
    )


def _empty_result_for_backend(clip: InputClip, built: BuiltBackend) -> BenchmarkResult:
    return BenchmarkResult(
        clip=clip.name,
        path=str(clip.path),
        private=clip.private,
        audio_s=len(clip.audio) / SR,
        engine=built.engine,
        device=built.device,
        status=built.status,
        walls=[],
        median_wall=_nan(),
        median_rtfx=_nan(),
        speakers=-1,
        providers=built.providers,
        reason=built.reason,
    )


def _time_built_backend(
    clip: InputClip,
    built: BuiltBackend,
    runs: int,
    warmup_sec: float,
) -> BenchmarkResult:
    audio_s = len(clip.audio) / SR
    if built.status != "ok" or built.run is None:
        return _empty_result_for_backend(clip, built)

    warmup_samples = max(0, int(SR * warmup_sec))
    try:
        built.run(clip.audio[:warmup_samples])
    except Exception as exc:  # noqa: BLE001
        return BenchmarkResult(
            clip=clip.name,
            path=str(clip.path),
            private=clip.private,
            audio_s=audio_s,
            engine=built.engine,
            device=built.device,
            status="fail",
            walls=[],
            median_wall=_nan(),
            median_rtfx=_nan(),
            speakers=-1,
            providers=built.providers,
            reason=f"warmup failed: {exc}",
        )

    walls: list[float] = []
    speakers = -1
    try:
        for _ in range(runs):
            start = time.perf_counter()
            speakers = built.run(clip.audio)
            walls.append(time.perf_counter() - start)
    except Exception as exc:  # noqa: BLE001
        return BenchmarkResult(
            clip=clip.name,
            path=str(clip.path),
            private=clip.private,
            audio_s=audio_s,
            engine=built.engine,
            device=built.device,
            status="fail",
            walls=walls,
            median_wall=_nan(),
            median_rtfx=_nan(),
            speakers=-1,
            providers=built.providers,
            reason=f"run failed: {exc}",
        )

    median_wall = statistics.median(walls)
    median_rtfx = audio_s / median_wall if median_wall else _nan()
    return BenchmarkResult(
        clip=clip.name,
        path=str(clip.path),
        private=clip.private,
        audio_s=audio_s,
        engine=built.engine,
        device=built.device,
        status="ok",
        walls=walls,
        median_wall=median_wall,
        median_rtfx=median_rtfx,
        speakers=speakers,
        providers=built.providers,
        reason="",
    )


def _time_backend(
    clip: InputClip,
    backend: BackendConfig,
    runs: int,
    warmup_sec: float,
) -> BenchmarkResult:
    return _time_built_backend(clip, _build_backend(backend), runs, warmup_sec)


def _metadata_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


def _mac_chip() -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:  # noqa: BLE001
        return ""
    return proc.stdout.strip()


def _collect_environment() -> EnvironmentInfo:
    try:
        import onnxruntime as ort

        ort_providers = ", ".join(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        ort_providers = f"unavailable ({exc})"

    try:
        import torch

        torch_version = getattr(torch, "__version__", "unknown")
        torch_mps = "available" if torch.backends.mps.is_available() else "unavailable"
    except Exception as exc:  # noqa: BLE001
        torch_version = f"unavailable ({exc})"
        torch_mps = "unavailable"

    return EnvironmentInfo(
        benchmark_date=time.strftime("%Y-%m-%d"),
        platform=platform.platform(),
        machine=platform.machine(),
        chip=_mac_chip(),
        python=sys.version.split()[0],
        onnxruntime=_metadata_version("onnxruntime"),
        onnxruntime_providers=ort_providers,
        torch=torch_version,
        torch_mps=torch_mps,
        pyannote_audio=_metadata_version("pyannote.audio"),
    )


def _render_markdown(
    env: EnvironmentInfo,
    inputs: list[InputSummary],
    results: list[BenchmarkResult],
    runs: int,
    warmup_sec: float,
) -> str:
    lines = [
        "# Speed: ONNX (this repo) vs pyannote.audio PyTorch",
        "",
        "RTFx = audio seconds / wall-clock seconds (higher is faster).",
        f"Each successful cell reports the median of {runs} timed runs after one {warmup_sec:.1f}s warmup.",
        "",
        "## Current Mac Benchmark Environment",
        "",
        "| field | value |",
        "|---|---|",
        f"| benchmark date | {_md_cell(env.benchmark_date)} |",
        f"| platform | {_md_cell(env.platform)} |",
        f"| machine | {_md_cell(env.machine)} |",
        f"| chip | {_md_cell(env.chip or 'unknown')} |",
        f"| python | {_md_cell(env.python)} |",
        f"| onnxruntime | {_md_cell(env.onnxruntime)} |",
        f"| onnxruntime providers | {_md_cell(env.onnxruntime_providers)} |",
        f"| torch | {_md_cell(env.torch)} |",
        f"| torch MPS | {_md_cell(env.torch_mps)} |",
        f"| pyannote.audio | {_md_cell(env.pyannote_audio)} |",
        "",
        "## Inputs",
        "",
        "| clip | audio s | source | path |",
        "|---|---:|---|---|",
    ]

    for item in inputs:
        source = "local/private" if item.private else "public/committed"
        lines.append(f"| {_md_cell(item.name)} | {item.audio_s:.1f} | {source} | {_md_cell(item.path)} |")

    if any(item.private for item in inputs):
        lines.extend(
            [
                "",
                "Local/private clips are used only as benchmark inputs and are not distributed with this repository.",
            ]
        )

    lines.extend(
        [
            "",
            "## Median Results",
            "",
            "| clip | audio s | engine | device | status | median wall s | median RTFx | speakers | providers |",
            "|---|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for result in results:
        lines.append(
            "| "
            f"{_md_cell(result.clip)} | "
            f"{result.audio_s:.1f} | "
            f"{_md_cell(result.engine)} | "
            f"{_md_cell(result.device)} | "
            f"{_md_cell(result.status)} | "
            f"{_format_float(result.median_wall, 1)} | "
            f"{_format_float(result.median_rtfx, 2)} | "
            f"{result.speakers} | "
            f"{_md_cell(result.providers or result.reason)} |"
        )

    lines.extend(
        [
            "",
            "## Raw Runs",
            "",
            "| clip | engine | device | status | wall s | reason |",
            "|---|---|---|---|---|---|",
        ]
    )
    for result in results:
        walls = ", ".join(_format_float(wall, 1) for wall in result.walls)
        lines.append(
            "| "
            f"{_md_cell(result.clip)} | "
            f"{_md_cell(result.engine)} | "
            f"{_md_cell(result.device)} | "
            f"{_md_cell(result.status)} | "
            f"{walls} | "
            f"{_md_cell(result.reason)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `ours` is this repo's torch-free ONNX Runtime diarization pipeline.",
            "- `pytorch` is the official `pyannote/speaker-diarization-community-1` pipeline.",
            "- CoreML runs with `CoreMLExecutionProvider, CPUExecutionProvider`; ONNX Runtime may fall back to CPU for unsupported nodes.",
            "- MPS runs with `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported PyTorch ops can fall back to CPU.",
            "",
        ]
    )
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="*", help="public/committed audio paths")
    parser.add_argument(
        "--private-audio",
        action="append",
        default=[],
        help="local/private audio path; may be passed more than once",
    )
    parser.add_argument("--runs", type=_positive_int, default=3, help="timed full-audio runs per cell")
    parser.add_argument("--warmup-sec", type=float, default=WARMUP_SEC, help="warmup audio seconds before timing")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="markdown output path")
    return parser.parse_args(argv)


def _load_audio(path: Path) -> np.ndarray:
    from pyannote_onnx_community.audio import load_audio

    return load_audio(path, SR)


def _load_inputs(public_paths: list[str], private_paths: list[str]) -> list[InputClip]:
    clips: list[InputClip] = []
    for path_text in public_paths:
        path = Path(path_text)
        clips.append(InputClip(path.name, path, False, _load_audio(path)))
    for index, path_text in enumerate(private_paths, start=1):
        path = Path(path_text)
        name = "long-clip" if len(private_paths) == 1 else f"long-clip-{index}"
        clips.append(InputClip(name, path, True, _load_audio(path)))
    return clips


def _summarize_inputs(clips: list[InputClip]) -> list[InputSummary]:
    return [
        InputSummary(
            name=clip.name,
            path="local/private" if clip.private else str(clip.path),
            private=clip.private,
            audio_s=len(clip.audio) / SR,
        )
        for clip in clips
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not args.audio and not args.private_audio:
        print("No audio provided. Pass one or more audio paths or --private-audio PATH.", file=sys.stderr)
        return 2

    clips = _load_inputs(args.audio, args.private_audio)
    inputs = _summarize_inputs(clips)
    results: list[BenchmarkResult] = []

    for backend in _build_backend_configs():
        built = _build_backend(backend)
        for clip in clips:
            result = _time_built_backend(clip, built, args.runs, args.warmup_sec)
            results.append(result)
            print(
                "ROW",
                result.clip,
                result.engine,
                result.device,
                result.status,
                _format_float(result.median_wall, 1),
                _format_float(result.median_rtfx, 2),
                result.speakers,
                result.providers or result.reason,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_markdown(_collect_environment(), inputs, results, args.runs, args.warmup_sec),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
