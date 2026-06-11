# Current Mac Benchmark Design

## Goal

Replace the existing README speed numbers with a reproducible benchmark based on the Apple M4 Pro Mac running this workspace. The benchmark should compare this repository's torch-free ONNX pipeline against the official pyannote.audio PyTorch community-1 pipeline across CPU and Apple accelerator backends, then rewrite the relevant benchmark/history commits into a clean sequence.

## Baseline Machine

The official benchmark baseline for this repo will be the Apple M4 Pro Mac running the Codex workspace. Benchmark output must record enough environment metadata to make the numbers interpretable:

- macOS version
- machine architecture and chip/model information when available
- Python version
- onnxruntime version and available execution providers
- torch version and MPS availability
- pyannote.audio version
- benchmark date

The private long clip source is only an audio source. It is not the benchmark machine.

## Benchmark Matrix

The benchmark matrix should keep the existing comparison shape:

| Engine | Backend | Purpose |
| --- | --- | --- |
| ours ONNX Runtime | CPUExecutionProvider | Portable torch-free CPU baseline |
| ours ONNX Runtime | CoreMLExecutionProvider + CPUExecutionProvider | Apple accelerator comparison |
| official pyannote.audio community-1 | CPU | Upstream CPU baseline |
| official pyannote.audio community-1 | MPS | Upstream Apple accelerator comparison |

Missing backends should be reported as skipped with a reason instead of silently omitted or represented by partial numbers.

## Inputs

The benchmark uses two audio inputs:

1. `tests/goldens/clip.wav`, the committed 60-second clip used by existing parity/e2e checks.
2. One local/private `long-clip` audio file selected from the 10-15 minute range.

The private `long-clip` file should not be committed to the repository. It may be copied to a local temporary path or referenced through a local path while running the benchmark. Public documentation should describe it only as a local/private long clip and include duration, but must not present it as a redistributable fixture.

## Measurement Method

For each input/backend pair:

1. Decode the audio once using the package audio loader.
2. Run one warmup inference on a short slice to absorb graph/kernel compilation and cache effects.
3. Run three measured full-audio inferences.
4. Report median wall-clock seconds and median RTFx.

RTFx is `audio_seconds / wall_clock_seconds`; higher is faster. The detailed results may include per-run wall times or min/max to show variance, but README should focus on the median values.

## Outputs

`docs/benchmark_results.md` should be the full benchmark record. It should include:

- benchmark environment metadata
- input audio metadata
- backend availability and skip reasons
- median results table
- raw per-run timings or compact min/max variance
- notes about CoreML/MPS behavior and private long-audio licensing

`README.md` should contain the short reader-facing speed section:

- a concise statement that the benchmark baseline is this Mac
- the main median results table
- the main interpretation: ONNX CPU is the portable torch-free win; PyTorch MPS may win when Apple GPU acceleration and torch are available
- a link or pointer to `docs/benchmark_results.md` for full methodology

## Git History Rewrite

After the benchmark tooling, results, and README updates are implemented and verified, rewrite the existing benchmark/README history rather than leaving a new pile of follow-up commits.

The intended cleanup is:

1. Create a backup tag such as `backup/main-before-benchmark-rewrite`.
2. Preserve commits that are about runtime behavior, tests, licensing, or other non-benchmark fixes unless they clearly belong in a benchmark/docs group.
3. Consolidate old speed-table/framing commits into a clean benchmark history, for example:
   - `feat(bench): add reproducible Apple M4 Pro benchmark matrix`
   - `docs: publish Apple M4 Pro benchmark results`
4. Verify the rewritten branch against the backup tag so only intended script/docs/README changes remain.
5. Run the selected test and benchmark sanity checks after the rewrite.
6. Keep the backup tag until the user confirms the rewritten history.

Because this repository currently has no configured remote/upstream, the rewrite does not need remote coordination during local cleanup. A later push to a remote would require force-push awareness.

## Verification

Verification should include:

- benchmark dry run or full run confirming available backends either produce numbers or explicit skip reasons
- `pytest tests/unit tests/parity` for torch-free repo sanity
- benchmark docs sanity check confirming README and `docs/benchmark_results.md` agree on headline numbers
- git history sanity check using `git log --oneline` and `git diff backup/main-before-benchmark-rewrite`

If PyTorch, pyannote.audio, Hugging Face credentials, CoreML, or MPS are unavailable in the current environment, the implementation should report the blocker clearly and avoid publishing incomplete numbers as final benchmark results.

## Risks And Decisions

- The private `long-clip` audio is suitable because a 10-15 minute long clip with a small speaker count matches a realistic diarization scenario.
- The long audio is not treated as redistributable content.
- Median of three measured runs is the chosen balance between stability and runtime cost.
- The Apple M4 Pro Mac is the only official benchmark machine for the replacement README numbers.
- The benchmark matrix intentionally includes Apple accelerator rows, but the README should frame CPU portability as the primary advantage.
