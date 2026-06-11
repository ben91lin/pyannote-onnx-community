import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_time_backend_reports_fail_for_unexpected_builder_error():
    bench = _load_benchmark_module()
    wav = np.zeros(bench.SR * 60, dtype=np.float32)

    def fail_builder():
        raise RuntimeError("broken model")

    result = bench._time_backend(
        clip=bench.InputClip(name="clip.wav", path=Path("clip.wav"), private=False, audio=wav),
        backend=bench.BackendConfig(engine="ours", device="cpu", builder=fail_builder),
        runs=3,
        warmup_sec=10.0,
    )

    assert result.status == "fail"
    assert result.reason == "build failed: broken model"
    assert result.median_wall != result.median_wall


def test_time_backend_reports_fail_for_warmup_error():
    bench = _load_benchmark_module()
    wav = np.zeros(bench.SR * 60, dtype=np.float32)

    def run(_wav):
        raise RuntimeError("warmup boom")

    result = bench._time_backend(
        clip=bench.InputClip(name="clip.wav", path=Path("clip.wav"), private=False, audio=wav),
        backend=bench.BackendConfig(engine="ours", device="cpu", builder=lambda: (run, "CPU")),
        runs=3,
        warmup_sec=10.0,
    )

    assert result.status == "fail"
    assert result.reason == "warmup failed: warmup boom"
    assert result.providers == "CPU"


def test_default_output_is_repository_relative():
    bench = _load_benchmark_module()
    args = bench._parse_args([])

    assert args.output == Path(__file__).resolve().parents[2] / "docs" / "benchmark_results.md"


def test_main_builds_each_backend_once_for_multiple_clips(monkeypatch, tmp_path):
    bench = _load_benchmark_module()
    build_calls = []
    run_calls = []
    perf_values = iter([0.0, 1.0, 10.0, 11.0])

    def run(wav):
        run_calls.append(len(wav))
        return 1

    def builder():
        build_calls.append("built")
        return run, "CPU"

    monkeypatch.setattr(
        bench,
        "_build_backend_configs",
        lambda: [bench.BackendConfig(engine="ours", device="cpu", builder=builder)],
    )
    monkeypatch.setattr(
        bench,
        "_load_inputs",
        lambda _public, _private: [
            bench.InputClip("a.wav", Path("a.wav"), False, np.zeros(bench.SR, dtype=np.float32)),
            bench.InputClip("b.wav", Path("b.wav"), False, np.zeros(bench.SR, dtype=np.float32)),
        ],
    )
    monkeypatch.setattr(bench, "_collect_environment", lambda: bench.EnvironmentInfo("", "", "", "", "", "", "", "", "", ""))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(perf_values))

    assert bench.main(["a.wav", "b.wav", "--runs", "1", "--output", str(tmp_path / "bench.md")]) == 0

    assert build_calls == ["built"]
    assert run_calls == [bench.SR, bench.SR, bench.SR, bench.SR]


def test_load_inputs_abstracts_private_audio_names(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr(bench, "_load_audio", lambda _path: np.zeros(bench.SR, dtype=np.float32))

    clips = bench._load_inputs(["tests/goldens/clip.wav"], ["/private/source/audio-name.ogg"])
    summaries = bench._summarize_inputs(clips)

    assert clips[0].name == "clip.wav"
    assert clips[1].name == "long-clip"
    assert summaries[1].name == "long-clip"
    assert summaries[1].path == "local/private"


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
        path="local/private",
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
        inputs=[bench.InputSummary("long-clip", "local/private", True, 600.0)],
        results=[result],
        runs=3,
        warmup_sec=10.0,
    )

    assert "Current Mac Benchmark Environment" in markdown
    assert "local/private" in markdown
    assert "Local/private clips are used only as benchmark inputs" in markdown
    assert "| long-clip | 600.0 | ours | cpu | ok | 33.0 | 18.18 | 3 | CPU |" in markdown
    assert "30.0, 33.0, 36.0" in markdown
    assert "RTFx = audio seconds / wall-clock seconds" in markdown


def test_render_markdown_escapes_pipe_and_newline_cells():
    bench = _load_benchmark_module()
    env = bench.EnvironmentInfo(
        benchmark_date="2026-06-10",
        platform="macOS|test\nhost",
        machine="arm64",
        chip="Apple M test",
        python="3.13.0",
        onnxruntime="1.0",
        onnxruntime_providers="CoreMLExecutionProvider|CPUExecutionProvider",
        torch="2.0",
        torch_mps="available",
        pyannote_audio="4.0.4",
    )
    result = bench.BenchmarkResult(
        clip="clip.wav",
        path="/tmp/clip.wav",
        private=False,
        audio_s=60.0,
        engine="ours",
        device="cpu",
        status="fail",
        walls=[],
        median_wall=float("nan"),
        median_rtfx=float("nan"),
        speakers=-1,
        providers="CPU|CoreML\nfallback",
        reason="bad|thing\nnext",
    )

    markdown = bench._render_markdown(
        env=env,
        inputs=[bench.InputSummary("clip.wav", "/tmp/a|b\nclip.wav", False, 60.0)],
        results=[result],
        runs=3,
        warmup_sec=10.0,
    )

    assert "macOS\\|test host" in markdown
    assert "/tmp/a\\|b clip.wav" in markdown
    assert "CPU\\|CoreML fallback" in markdown
    assert "bad\\|thing next" in markdown
