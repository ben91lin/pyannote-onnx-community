# Current Mac Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing speed benchmark with a reproducible Apple M4 Pro benchmark matrix and rewrite the old benchmark/README history into a clean sequence.

**Architecture:** Keep the benchmark as a standalone script in `scripts/benchmark.py`, but make it testable by splitting timing, environment collection, backend selection, and markdown rendering into small helper functions. The script writes the full benchmark record to `docs/benchmark_results.md`; `README.md` keeps the short headline table and links to the full methodology.

**Tech Stack:** Python 3.11+, argparse, dataclasses, importlib.metadata, platform/subprocess for environment metadata, onnxruntime providers, pyannote.audio + torch for dev-only upstream comparison, pytest for script helper tests, git interactive rebase for final history cleanup.

---

## File Structure

- Modify `scripts/benchmark.py`
  - Add CLI flags for `--runs`, `--warmup-sec`, `--output`, and repeatable `--private-audio`.
  - Add dataclasses for input clips, backend configs, benchmark results, skip records, and environment metadata.
  - Report all four intended backends: ONNX CPU, ONNX CoreML, PyTorch CPU, PyTorch MPS.
  - Write median-of-three benchmark results and skip reasons to markdown.
- Create `tests/unit/test_benchmark_script.py`
  - Unit-test helper behavior without loading real ONNX/PyTorch models.
  - Cover median timing, skip rows, markdown rendering, and private input labeling.
- Modify `docs/benchmark_results.md`
  - Replace single warmed-run table with Apple M4 Pro environment, input metadata, median table, and raw timing detail.
- Modify `README.md`
  - Replace old speed numbers with the Apple M4 Pro median table and concise interpretation.
- No audio fixture is added to the repository.
  - The private `long-clip` input is copied or referenced locally while benchmarking, then left untracked.

---

### Task 1: Add Unit Tests For Benchmark Helpers

**Files:**
- Create: `tests/unit/test_benchmark_script.py`
- Modify: none
- Test: `tests/unit/test_benchmark_script.py`

- [ ] **Step 1: Create the benchmark script test file**

Add this file at `tests/unit/test_benchmark_script.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_time_backend_reports_median_wall_and_rtfx(monkeypatch):
    bench = _load_benchmark_module()
    calls = []

    def fake_perf_counter():
        values = iter([0.0, 2.0, 10.0, 13.0, 20.0, 24.0])

        def _next():
            return next(values)

        return _next

    monkeypatch.setattr(bench.time, "perf_counter", fake_perf_counter())

    def run(wav):
        calls.append(len(wav))
        return 2

    wav = np.zeros(bench.SR * 60, dtype=np.float32)
    result = bench._time_backend(
        clip=bench.InputClip(name="clip.wav", path=Path("clip.wav"), private=False, audio=wav),
        backend=bench.BackendConfig(
            engine="ours",
            device="cpu",
            builder=lambda: (run, "CPU"),
        ),
        runs=3,
        warmup_sec=10.0,
    )

    assert calls[0] == bench.SR * 10
    assert calls[1:] == [bench.SR * 60, bench.SR * 60, bench.SR * 60]
    assert result.status == "ok"
    assert result.walls == [2.0, 3.0, 4.0]
    assert result.median_wall == 3.0
    assert result.median_rtfx == 20.0
    assert result.speakers == 2
    assert result.providers == "CPU"


def test_time_backend_returns_skip_when_builder_fails():
    bench = _load_benchmark_module()
    wav = np.zeros(bench.SR * 60, dtype=np.float32)

    def fail_builder():
        raise bench.BackendUnavailable("CoreMLExecutionProvider unavailable")

    result = bench._time_backend(
        clip=bench.InputClip(name="private-long.ogg", path=Path("/tmp/private-long.ogg"), private=True, audio=wav),
        backend=bench.BackendConfig(engine="ours", device="coreml", builder=fail_builder),
        runs=3,
        warmup_sec=10.0,
    )

    assert result.status == "skip"
    assert result.reason == "CoreMLExecutionProvider unavailable"
    assert result.private is True
    assert result.median_wall != result.median_wall
    assert result.median_rtfx != result.median_rtfx


def test_render_markdown_includes_environment_private_inputs_and_raw_runs():
    bench = _load_benchmark_module()
    env = bench.EnvironmentInfo(
        benchmark_date="2026-06-10",
        platform="macOS test",
        machine="arm64",
        chip="Apple M test",
        python="3.13.0",
        onnxruntime="1.0",
        onnxruntime_providers="CoreMLExecutionProvider, CPUExecutionProvider",
        torch="2.0",
        torch_mps="available",
        pyannote_audio="4.0.4",
    )
    result = bench.BenchmarkResult(
        clip="long-clip",
        path="/tmp/long-clip",
        private=True,
        audio_s=600.0,
        engine="ours",
        device="cpu",
        status="ok",
        walls=[30.0, 33.0, 36.0],
        median_wall=33.0,
        median_rtfx=18.18,
        speakers=3,
        providers="CPU",
        reason="",
    )

    markdown = bench._render_markdown(
        env=env,
        inputs=[bench.InputSummary("long-clip", "/tmp/long-clip", True, 600.0)],
        results=[result],
        runs=3,
        warmup_sec=10.0,
    )

    assert "Current Mac Benchmark Environment" in markdown
    assert "local/private" in markdown
    assert "| long-clip | 600.0 | ours | cpu | ok | 33.0 | 18.18 | 3 | CPU |" in markdown
    assert "30.0, 33.0, 36.0" in markdown
    assert "RTFx = audio seconds / wall-clock seconds" in markdown
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest tests/unit/test_benchmark_script.py -v
```

Expected: FAIL because `InputClip`, `BackendConfig`, `BackendUnavailable`, `_time_backend`, `EnvironmentInfo`, `BenchmarkResult`, `InputSummary`, and `_render_markdown` do not exist yet in `scripts/benchmark.py`.

### Task 2: Refactor Benchmark Script For Median Runs And Metadata

**Files:**
- Modify: `scripts/benchmark.py`
- Test: `tests/unit/test_benchmark_script.py`

- [ ] **Step 1: Replace `scripts/benchmark.py` with a testable benchmark script**

Replace the file contents with this implementation:

```python
"""Speed comparison: our ONNX pipeline vs upstream pyannote.audio PyTorch.

Device matrix on the Apple M4 Pro:
    ours ONNX {CPU, CoreML}  x  pytorch {CPU, MPS}

Run in the dev venv (needs torch + pyannote.audio; onnxruntime must have CoreML for
the CoreML row):
    python scripts/benchmark.py tests/goldens/clip.wav --private-audio /tmp/long.ogg

Each input is decoded once via the package's PyAV loader and run through every
backend. Each backend gets one warmup inference on a short slice before three
measured full-audio runs. RTFx = audio_seconds / wall_seconds (higher is faster).
"""

from __future__ import annotations

import argparse
import math
import os

# Must be set before torch imports so MPS-unsupported ops fall back to CPU rather than error.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import date  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import median  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402

from pyannote_onnx_community.audio import load_audio  # noqa: E402

SR = 16000
DEFAULT_RUNS = 3
DEFAULT_WARMUP_SEC = 10.0


class BackendUnavailable(RuntimeError):
    """Raised when a benchmark backend cannot be built on this machine."""


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


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _mac_chip() -> str:
    if platform.system() != "Darwin":
        return "n/a"
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        out = ""
    if out:
        return out
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chip:"):
            return stripped.split(":", 1)[1].strip()
    return "unknown"


def _collect_environment() -> EnvironmentInfo:
    try:
        import onnxruntime as ort

        ort_version = ort.__version__
        ort_providers = ", ".join(ort.get_available_providers())
    except Exception as e:  # noqa: BLE001
        ort_version = f"unavailable: {e}"
        ort_providers = "unavailable"

    try:
        import torch

        torch_version = torch.__version__
        torch_mps = "available" if torch.backends.mps.is_available() else "unavailable"
    except Exception as e:  # noqa: BLE001
        torch_version = f"unavailable: {e}"
        torch_mps = "unavailable"

    return EnvironmentInfo(
        benchmark_date=date.today().isoformat(),
        platform=platform.platform(),
        machine=platform.machine(),
        chip=_mac_chip(),
        python=sys.version.split()[0],
        onnxruntime=ort_version,
        onnxruntime_providers=ort_providers,
        torch=torch_version,
        torch_mps=torch_mps,
        pyannote_audio=_version("pyannote.audio"),
    )


def _ours_runner(providers: list[str]) -> tuple[Callable[[np.ndarray], int], str]:
    from pyannote_onnx_community import ONNXSpeakerDiarization

    dia = ONNXSpeakerDiarization(providers=providers)
    try:
        actual = ",".join(p.replace("ExecutionProvider", "") for p in dia._seg.get_providers())
    except Exception:  # noqa: BLE001
        actual = ",".join(p.replace("ExecutionProvider", "") for p in providers)

    def run(wav: np.ndarray) -> int:
        return len(dia(wav).labels())

    return run, actual


def _pytorch_runner(device: str) -> tuple[Callable[[np.ndarray], int], str]:
    import torch
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=get_token())
    pipe.to(torch.device(device))

    def run(wav: np.ndarray) -> int:
        out = pipe({"waveform": torch.from_numpy(wav[np.newaxis, :]), "sample_rate": SR})
        ann = getattr(out, "speaker_diarization", out)
        return len(ann.labels())

    return run, device


def _require_coreml() -> None:
    import onnxruntime as ort

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise BackendUnavailable("CoreMLExecutionProvider unavailable")


def _require_mps() -> None:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise BackendUnavailable(f"torch unavailable: {e}") from e
    if not torch.backends.mps.is_available():
        raise BackendUnavailable("torch MPS unavailable")


def _build_configs() -> list[BackendConfig]:
    return [
        BackendConfig("ours", "cpu", lambda: _ours_runner(["CPUExecutionProvider"])),
        BackendConfig(
            "ours",
            "coreml",
            lambda: (_require_coreml() or _ours_runner(["CoreMLExecutionProvider", "CPUExecutionProvider"])),
        ),
        BackendConfig("pytorch", "cpu", lambda: _pytorch_runner("cpu")),
        BackendConfig("pytorch", "mps", lambda: (_require_mps() or _pytorch_runner("mps"))),
    ]


def _time_backend(clip: InputClip, backend: BackendConfig, runs: int, warmup_sec: float) -> BenchmarkResult:
    dur = len(clip.audio) / SR
    try:
        run, actual = backend.builder()
    except Exception as e:  # noqa: BLE001
        return BenchmarkResult(
            clip=clip.name,
            path=str(clip.path),
            private=clip.private,
            audio_s=dur,
            engine=backend.engine,
            device=backend.device,
            status="skip",
            walls=[],
            median_wall=math.nan,
            median_rtfx=math.nan,
            speakers=-1,
            providers="",
            reason=str(e),
        )

    warm = clip.audio[: int(SR * warmup_sec)] if clip.audio.shape[0] > int(SR * warmup_sec) else clip.audio
    try:
        run(warm)
    except Exception as e:  # noqa: BLE001
        return BenchmarkResult(
            clip=clip.name,
            path=str(clip.path),
            private=clip.private,
            audio_s=dur,
            engine=backend.engine,
            device=backend.device,
            status="skip",
            walls=[],
            median_wall=math.nan,
            median_rtfx=math.nan,
            speakers=-1,
            providers=actual,
            reason=f"warmup failed: {e}",
        )

    walls: list[float] = []
    speakers = -1
    try:
        for _ in range(runs):
            t0 = time.perf_counter()
            speakers = run(clip.audio)
            walls.append(time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        return BenchmarkResult(
            clip=clip.name,
            path=str(clip.path),
            private=clip.private,
            audio_s=dur,
            engine=backend.engine,
            device=backend.device,
            status="fail",
            walls=walls,
            median_wall=math.nan,
            median_rtfx=math.nan,
            speakers=speakers,
            providers=actual,
            reason=str(e),
        )

    median_wall = float(median(walls))
    return BenchmarkResult(
        clip=clip.name,
        path=str(clip.path),
        private=clip.private,
        audio_s=dur,
        engine=backend.engine,
        device=backend.device,
        status="ok",
        walls=walls,
        median_wall=median_wall,
        median_rtfx=dur / median_wall if median_wall else math.nan,
        speakers=speakers,
        providers=actual,
        reason="",
    )


def _format_float(value: float, digits: int = 1) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


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
        "RTFx = audio seconds / wall-clock seconds (higher = faster).",
        f"Each backend uses one {warmup_sec:.1f}s warmup run followed by {runs} measured full-audio runs.",
        "Tables report median wall time and median RTFx.",
        "",
        "## Current Mac Benchmark Environment",
        "",
        f"- Date: `{env.benchmark_date}`",
        f"- Platform: `{env.platform}`",
        f"- Machine: `{env.machine}`",
        f"- Chip: `{env.chip}`",
        f"- Python: `{env.python}`",
        f"- onnxruntime: `{env.onnxruntime}`",
        f"- ONNX Runtime providers: `{env.onnxruntime_providers}`",
        f"- torch: `{env.torch}`",
        f"- torch MPS: `{env.torch_mps}`",
        f"- pyannote.audio: `{env.pyannote_audio}`",
        "",
        "## Inputs",
        "",
        "| clip | audio s | visibility | path |",
        "|------|--------:|------------|------|",
    ]
    for item in inputs:
        visibility = "local/private" if item.private else "committed"
        lines.append(f"| {item.name} | {item.audio_s:.1f} | {visibility} | `{item.path}` |")

    lines.extend(
        [
            "",
            "The local/private long audio is used only to produce the maintainer benchmark numbers.",
            "It is not committed to this repository and is not distributed as a fixture.",
            "",
            "## Median Results",
            "",
            "| clip | audio s | engine | device | status | median wall s | median RTFx | speakers | providers |",
            "|------|--------:|--------|--------|--------|--------------:|------------:|---------:|-----------|",
        ]
    )
    for row in results:
        detail = row.providers if row.status == "ok" else row.reason
        lines.append(
            "| "
            f"{row.clip} | {row.audio_s:.1f} | {row.engine} | {row.device} | {row.status} | "
            f"{_format_float(row.median_wall, 1)} | {_format_float(row.median_rtfx, 2)} | "
            f"{row.speakers if row.speakers >= 0 else 'n/a'} | {detail} |"
        )

    lines.extend(
        [
            "",
            "## Raw Runs",
            "",
            "| clip | engine | device | wall runs s |",
            "|------|--------|--------|-------------|",
        ]
    )
    for row in results:
        walls = ", ".join(f"{v:.1f}" for v in row.walls) if row.walls else row.reason
        lines.append(f"| {row.clip} | {row.engine} | {row.device} | {walls} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `ours` is the torch-free ONNX Runtime pipeline in this repository.",
            "- `pytorch` is the official `pyannote/speaker-diarization-community-1` pipeline.",
            "- CoreML may run unsupported dynamic-shape nodes on CPU even when CoreML is listed as a provider.",
            "- MPS uses `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported PyTorch ops can fall back to CPU.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_inputs(public_audio: list[str], private_audio: list[str]) -> list[InputClip]:
    clips: list[InputClip] = []
    for raw, private in [(p, False) for p in public_audio] + [(p, True) for p in private_audio]:
        path = Path(raw).expanduser()
        wav = load_audio(str(path), SR)
        clips.append(InputClip(name=path.name, path=path, private=private, audio=wav))
    return clips


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="*", help="Committed or redistributable benchmark audio paths.")
    parser.add_argument(
        "--private-audio",
        action="append",
        default=[],
        help="Local/private benchmark audio path. It is marked private in generated docs.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Measured full-audio runs per backend.")
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=DEFAULT_WARMUP_SEC,
        help="Warmup slice length in seconds.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "benchmark_results.md"),
        help="Markdown output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if not args.audio and not args.private_audio:
        raise SystemExit("provide at least one audio path")

    clips = _load_inputs(args.audio, args.private_audio)
    summaries = [
        InputSummary(clip.name, str(clip.path), clip.private, len(clip.audio) / SR)
        for clip in clips
    ]
    results: list[BenchmarkResult] = []
    for backend in _build_configs():
        for clip in clips:
            result = _time_backend(clip=clip, backend=backend, runs=args.runs, warmup_sec=args.warmup_sec)
            results.append(result)
            print(
                "ROW",
                result.clip,
                result.engine,
                result.device,
                result.status,
                _format_float(result.median_wall, 1),
                _format_float(result.median_rtfx, 2),
                result.reason,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _render_markdown(
            env=_collect_environment(),
            inputs=summaries,
            results=results,
            runs=args.runs,
            warmup_sec=args.warmup_sec,
        )
        + "\n"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the benchmark helper tests**

Run:

```bash
pytest tests/unit/test_benchmark_script.py -v
```

Expected: PASS for all three tests.

- [ ] **Step 3: Run a fast script sanity check**

Run:

```bash
python scripts/benchmark.py --help
```

Expected: command exits 0 and shows `--runs`, `--warmup-sec`, `--output`, and `--private-audio`.

- [ ] **Step 4: Commit the benchmark script refactor and tests**

Run:

```bash
git add scripts/benchmark.py tests/unit/test_benchmark_script.py
git commit -m "feat(bench): add reproducible median benchmark script"
```

Expected: commit includes the script refactor and the now-passing helper tests.

---

### Task 3: Select A Private `long-clip` Candidate

**Files:**
- Modify: none
- Test: local shell commands only

- [ ] **Step 1: Find 10-15 minute candidates from a private audio source**

Use file size as the first-pass filter, then validate only a small number of
local candidates by decoding. Do not run duration probing over the entire
private corpus.

Expected: one private candidate that decodes to a duration between 600.0 and
900.0 seconds.

- [ ] **Step 2: Copy one candidate to a local temporary benchmark directory**

Copy the chosen candidate locally:

```bash
mkdir -p /tmp/pyannote-onnx-benchmark
# Copy the private candidate here without committing it to the repository.
cp /path/to/private/candidate /tmp/pyannote-onnx-benchmark/long-clip
```

Expected: `/tmp/pyannote-onnx-benchmark/long-clip` exists locally and is not inside the repository.

- [ ] **Step 3: Confirm the local private audio decodes**

Run:

```bash
python - <<'PY'
from pyannote_onnx_community.audio import load_audio
wav = load_audio("/tmp/pyannote-onnx-benchmark/long-clip", 16000)
print(round(len(wav) / 16000, 1), wav.dtype, wav.shape)
PY
```

Expected: duration prints between `600.0` and `900.0`, dtype is float-like, and shape is one-dimensional.

---

### Task 4: Run The Current-Mac Benchmark

**Files:**
- Modify: `docs/benchmark_results.md`
- Test: generated benchmark output

- [ ] **Step 1: Confirm dev-only benchmark dependencies are available**

Run:

```bash
python - <<'PY'
import importlib
for name in ["onnxruntime", "torch", "pyannote.audio", "pyannote.metrics"]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: unavailable: {exc}")
    else:
        print(f"{name}: ok")
PY
```

Expected: `onnxruntime`, `torch`, and `pyannote.audio` are `ok`. `pyannote.metrics` is not required for the benchmark script, but if it is unavailable, note that e2e DER tests are not runnable.

- [ ] **Step 2: Confirm Hugging Face token availability for upstream PyTorch**

Run:

```bash
python - <<'PY'
from huggingface_hub import get_token
print("HF token:", "available" if get_token() else "missing")
PY
```

Expected: `HF token: available`. If missing, export `HF_TOKEN` before running the benchmark or expect the PyTorch rows to be skipped/fail.

- [ ] **Step 3: Run the full benchmark on the Apple M4 Pro**

Run:

```bash
python scripts/benchmark.py \
  tests/goldens/clip.wav \
  --private-audio /tmp/pyannote-onnx-benchmark/long-clip \
  --runs 3 \
  --warmup-sec 10 \
  --output docs/benchmark_results.md
```

Expected: the command writes `docs/benchmark_results.md`. Rows for available backends should print `ok`; unavailable backends should print `skip` with a reason.

- [ ] **Step 4: Inspect the generated benchmark document**

Run:

```bash
sed -n '1,220p' docs/benchmark_results.md
```

Expected:
- The environment section identifies the Apple M4 Pro.
- The inputs section marks `tests/goldens/clip.wav` as committed.
- The inputs section marks `long-clip` as `local/private`.
- The median table contains rows for ONNX CPU, ONNX CoreML, PyTorch CPU, and PyTorch MPS, either `ok` or explicit `skip`.
- The raw runs table has three wall-clock values for each successful row.

- [ ] **Step 5: Commit the benchmark results**

Run:

```bash
git add docs/benchmark_results.md
git commit -m "docs(bench): record Apple M4 Pro benchmark results"
```

Expected: commit contains only the regenerated benchmark results document.

---

### Task 5: Update README Speed Section

**Files:**
- Modify: `README.md`
- Test: README/docs consistency check

- [ ] **Step 1: Replace the README Speed section with Apple M4 Pro median results**

Edit `README.md` section `## Speed` so it has this shape, replacing the example numeric values with the actual medians from `docs/benchmark_results.md`:

```markdown
## Speed

**Core advantage: fast on plain CPU, with zero `torch` / GPU / accelerator
dependency.** The benchmark below is measured on the Apple M4 Pro benchmark
machine. Each backend gets one 10s warmup run followed by
three measured full-audio runs; tables report the median. Full environment,
private-input notes, raw timings, and skip reasons are in
`docs/benchmark_results.md`.

| clip | audio s | engine | device | median wall s | median RTFx | speakers |
|------|--------:|--------|--------|--------------:|------------:|---------:|
| clip.wav | ACTUAL | ours | cpu | ACTUAL | ACTUAL | ACTUAL |
| clip.wav | ACTUAL | ours | coreml | ACTUAL | ACTUAL | ACTUAL |
| clip.wav | ACTUAL | pytorch | cpu | ACTUAL | ACTUAL | ACTUAL |
| clip.wav | ACTUAL | pytorch | mps | ACTUAL | ACTUAL | ACTUAL |
| long-clip | ACTUAL | ours | cpu | ACTUAL | ACTUAL | ACTUAL |
| long-clip | ACTUAL | ours | coreml | ACTUAL | ACTUAL | ACTUAL |
| long-clip | ACTUAL | pytorch | cpu | ACTUAL | ACTUAL | ACTUAL |
| long-clip | ACTUAL | pytorch | mps | ACTUAL | ACTUAL | ACTUAL |

The portable headline is the ONNX CPU row: it needs no CUDA, no MPS, no CoreML,
and no `torch`. Accelerator rows map the Apple Silicon landscape. Where Apple
GPU acceleration and the multi-GB PyTorch dependency are both acceptable,
PyTorch-MPS may be faster on raw throughput; this repo's win is the deployment
surface while staying fast on CPU.
```

If any backend is skipped, omit that skipped row from the README headline table and mention the skip in one sentence pointing to `docs/benchmark_results.md`.

- [ ] **Step 2: Verify README and benchmark docs agree**

Run:

```bash
python - <<'PY'
from pathlib import Path
readme = Path("README.md").read_text()
bench = Path("docs/benchmark_results.md").read_text()
required = ["clip.wav", "long-clip", "median", "docs/benchmark_results.md"]
for text in required:
    assert text in readme, f"missing from README: {text}"
assert "Current Mac Benchmark Environment" in bench
assert "local/private" in bench
print("README/docs benchmark sanity ok")
PY
```

Expected: `README/docs benchmark sanity ok`.

- [ ] **Step 3: Commit the README update**

Run:

```bash
git add README.md
git commit -m "docs: publish Apple M4 Pro benchmark headline"
```

Expected: commit contains only the README speed-section update.

---

### Task 6: Run Repo Sanity Tests

**Files:**
- Modify: none
- Test: existing unit and parity suites

- [ ] **Step 1: Run torch-free tests**

Run:

```bash
pytest tests/unit tests/parity -v
```

Expected: PASS. If parity requires cached model artifacts that are unavailable, record the exact skip/failure reason in the implementation summary and at least keep `pytest tests/unit -v` passing.

- [ ] **Step 2: Run benchmark-specific tests again**

Run:

```bash
pytest tests/unit/test_benchmark_script.py -v
```

Expected: PASS.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: no uncommitted changes.

---

### Task 7: Rewrite Benchmark/README Git History

**Files:**
- Modify: git history only
- Test: `git diff` against backup tag, log inspection, tests

- [ ] **Step 1: Create a backup tag**

Run:

```bash
git tag backup/main-before-benchmark-rewrite main
git tag -l 'backup/*'
```

Expected: output includes `backup/main-before-benchmark-rewrite`.

- [ ] **Step 2: Inspect the commits to rewrite**

Run:

```bash
git log --oneline --reverse c9d92a2^..HEAD
```

Expected: output includes the old benchmark/README commits:
- `c9d92a2 feat: speed benchmark vs upstream PyTorch + captured CPU results (~22-24x faster)`
- `2cad235 docs: README with positioning, usage, validation + speed tables`
- `ef192d4 feat(bench): add GPU device matrix (ours CoreML / pytorch MPS) + honest speed framing`
- `9ab5147 docs: lead Speed section with CPU/torch-free portability advantage`
- the new plan/spec/test/script/docs commits from this work

- [ ] **Step 3: Generate the non-interactive rebase todo**

Run this command to classify known benchmark/spec/plan/docs commits as `fixup` lines and leave unrelated runtime/test/license commits as `pick` lines:

```bash
python - <<'PY'
import subprocess
from pathlib import Path

benchmark_subjects = {
    "docs: README with positioning, usage, validation + speed tables",
    "feat(bench): add GPU device matrix (ours CoreML / pytorch MPS) + honest speed framing",
    "docs: lead Speed section with CPU/torch-free portability advantage",
    "docs: add Apple M4 Pro benchmark design",
    "docs: add Apple M4 Pro benchmark implementation plan",
    "feat(bench): add reproducible median benchmark script",
    "docs(bench): record Apple M4 Pro benchmark results",
    "docs: publish Apple M4 Pro benchmark headline",
}
first_benchmark = "c9d92a2"
raw = subprocess.check_output(
    ["git", "log", "--format=%H%x00%s", "--reverse", f"{first_benchmark}^..HEAD"],
    text=True,
)

lines = []
for row in raw.splitlines():
    commit, subject = row.split("\x00", 1)
    short = commit[:7]
    if short == first_benchmark:
        action = "pick"
    elif subject in benchmark_subjects:
        action = "fixup"
    else:
        action = "pick"
    lines.append(f"{action} {commit} {subject}")

todo = "\n".join(lines) + "\n"
Path("/tmp/pyannote-benchmark-rebase-todo.txt").write_text(todo)
print(todo)
PY
```

Expected: `/tmp/pyannote-benchmark-rebase-todo.txt` starts with `pick c9d92a2...` and uses `fixup` for benchmark/spec/plan/result/README commits. The todo is in chronological order because it comes directly from `git log --reverse`.

- [ ] **Step 4: Run the non-interactive rebase**

Run:

```bash
cat > /tmp/pyannote-benchmark-rebase-editor.sh <<'SCRIPT'
#!/bin/sh
cp /tmp/pyannote-benchmark-rebase-todo.txt "$1"
SCRIPT
chmod +x /tmp/pyannote-benchmark-rebase-editor.sh
GIT_SEQUENCE_EDITOR=/tmp/pyannote-benchmark-rebase-editor.sh git rebase -i c9d92a2^
```

Expected: rebase completes. If conflicts occur, inspect each conflicted file manually with `git diff --name-only --diff-filter=U`, resolve the intended final content, `git add` the files, then `git rebase --continue`.

- [ ] **Step 5: Reword the squashed benchmark commit**

Run:

```bash
git log --oneline -8
```

Identify the new squashed benchmark commit hash, then run:

```bash
git commit --amend -m "feat(bench): add reproducible Apple M4 Pro benchmark matrix"
```

Expected: the benchmark commit message is clean and includes the final script, benchmark docs, README headline, and superpowers spec/plan files.

- [ ] **Step 6: Verify content against the backup tag**

Run:

```bash
git diff --stat backup/main-before-benchmark-rewrite
git diff -- docs/benchmark_results.md README.md scripts/benchmark.py tests/unit/test_benchmark_script.py docs/superpowers
```

Expected: diffs show only the intended benchmark/script/docs/spec/plan changes. No unrelated runtime files should differ unexpectedly.

- [ ] **Step 7: Run final sanity checks after rewrite**

Run:

```bash
pytest tests/unit/test_benchmark_script.py -v
pytest tests/unit -v
git status --short
```

Expected: benchmark helper tests pass, unit tests pass, and the worktree is clean.

- [ ] **Step 8: Leave backup tag in place for user confirmation**

Run:

```bash
git tag -l 'backup/*'
```

Expected: `backup/main-before-benchmark-rewrite` still exists. Do not delete it until the user reviews the rewritten history.

---

## Self-Review Notes

- Spec coverage: the plan covers Apple M4 Pro environment metadata, CPU/CoreML/PyTorch CPU/MPS backend matrix, committed plus private `long-clip` inputs, warmup plus three measured runs, markdown outputs, README update, verification, and git history rewrite with backup tag.
- Placeholder scan: the plan avoids future commit hash placeholders by generating the rebase todo from `git log --reverse` and exact commit subjects.
- Type consistency: test names and helper dataclasses match the implementation proposed for `scripts/benchmark.py`.
