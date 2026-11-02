# Concurrent Runtime Implementation Plan

Spec: `docs/superpowers/specs/2026-11-02-concurrent-runtime-design.md`

**Goal:** Add configurable thread limits, lazy fork-safe sessions, and a shared PLDA, without changing existing API behaviour or diarization output.

**Architecture:** `SessionConfig` is an immutable value in `config.py`. `_lib.py` owns one locked, process-wide session cache and a fork handler. Public classes hold a handle and fetch sessions per call.

**Tech Stack:** Python 3.11+, onnxruntime, pytest. `onnx` is added as a test-only dependency to build a tiny model.

**Baseline in a network-restricted environment:** 104 passed, 7 failed, 2 skipped. The 7 failures are `tests/unit/test_diarization.py` trying to download PLDA. Task 4 fixes them.

---

## File Structure

| File | Change |
| --- | --- |
| `pyannote_onnx_community/config.py` | Add `SessionConfig` |
| `pyannote_onnx_community/_lib.py` | Unified session cache, fork handler, session handle |
| `pyannote_onnx_community/_plda.py` | Add `load_plda` |
| `pyannote_onnx_community/_pipeline.py` | `_load_plda` uses `load_plda` |
| `pyannote_onnx_community/diarization.py` | New arguments, lazy sessions, `preload()` |
| `pyannote_onnx_community/vad.py` | New argument, lazy session, `preload()` |
| `pyannote_onnx_community/embedding.py` | New argument, lazy session, `preload()` |
| `pyannote_onnx_community/__init__.py` | Export `SessionConfig`, `clear_session_cache` |
| `pyproject.toml` | Add `test` extra |
| `tests/unit/test_session_config.py` | New |
| `tests/unit/test_session_cache.py` | New |
| `tests/unit/test_lib_providers.py` | Use `clear_session_cache` |
| `tests/unit/test_diarization.py` | Stub PLDA loading |
| `tests/unit/test_lazy_sessions.py` | New |
| `tests/integration/test_concurrency.py` | New; subprocess tests against a real tiny model |
| `README.md` | Usage and deployment guidance |

Work on a branch such as `feat/concurrent-runtime`. Use one commit per task.

---

### Task 1: `SessionConfig`

**Files:** Modify `config.py`; create `tests/unit/test_session_config.py`.

- Write failing tests for defaults, provider-option hashing and round-trip, option-order independence, idempotent freezing, invalid thread counts (`-1`, `1.5`, `True`, `"2"`), empty providers, and `from_env` (set, blank, unset with `providers=`, invalid).
- Implement `SessionConfig` as a frozen dataclass whose `__post_init__` freezes `(name, options)` pairs into sorted tuples and validates `intra_op_threads`; `ort_providers()` converts back; `from_env(prefix)` reads `{prefix}_INTRA_OP_THREADS`.
- Run `pytest tests/unit/test_session_config.py tests/unit/test_config.py -q` and commit.

### Task 2: Unified Session Cache

**Files:** Modify `_lib.py`, `tests/unit/test_lib_providers.py`; create `tests/unit/test_session_cache.py`.

- Replace `_seg_cache` / `_emb_cache` with one `_session_cache` keyed by `(model_path, SessionConfig)` behind `_session_lock`.
- `get_session(path, config)` holds the lock across lookup and creation; `_new_session` sets `intra_op_num_threads`.
- Add `clear_session_cache()`. `load_segmentation_session` / `load_embedding_session` accept `providers` (including option pairs) and an optional `session_config`.
- Fork handler (`os.register_at_fork(after_in_child=...)`, guarded by `hasattr`): replace the lock, `Py_IncRef` every cached session, clear the cache.
- Tests: equal configs share, different threads/providers do not, threads reach `SessionOptions`, provider options reach the session, legacy loaders accept options, clearing works, the fork hook leaks, clears, and replaces the lock.
- Update `test_lib_providers.py` to use `clear_session_cache()`.

### Task 3: Session Handle

**Files:** Modify `_lib.py`; add to `tests/unit/test_session_cache.py`.

- `SessionHandle(path, config, session)` (frozen) with `get()`; `make_session_handle(repo_or_path, *, session, session_config, providers)` resolves the model path now and builds later; injected session plus config raises `ValueError`; config beats providers.

### Task 4: Shared PLDA

**Files:** Modify `_plda.py`, `_pipeline.py`, `tests/unit/test_diarization.py`.

- `load_plda(repo_id)` caches per process under a lock; a `None` result is not cached; the lock is replaced after fork.
- `PyannoteOnnxClient._load_plda` uses `load_plda` (lazy import kept, explicit `plda=` keeps priority, `RuntimeError` on `None` kept).
- Stub `load_plda` in `tests/unit/test_diarization.py`'s `_make_dia`, removing the network dependency behind the 7 baseline failures.
- Cache test: two loads load once; `None` is not cached.

### Task 5: Lazy Public Classes

**Files:** Modify `diarization.py`, `vad.py`, `embedding.py`, `__init__.py`; create `tests/unit/test_lazy_sessions.py`.

- Constructors build handles (`make_session_handle`) instead of sessions; `_seg` / `_emb` become properties; `preload()` builds sessions (and PLDA for diarization).
- `ONNXSpeakerDiarization.__call__` builds `PyannoteOnnxClient` per call.
- New arguments go after existing ones so positional callers are unaffected.
- Export `SessionConfig` and `clear_session_cache`.
- Tests: no session at construction; `preload()` builds once; VAD and diarization with equal configs share one segmentation session; `providers` and `seg_session_config` reach the session; injected session plus config raises; `dia._seg` returns the live session.
- Run `pytest tests/unit tests/parity -q`: no failures expected.

### Task 6: Real-Runtime Concurrency Tests

**Files:** Modify `pyproject.toml`; create `tests/integration/__init__.py` and `tests/integration/test_concurrency.py`.

- Add a torch-free `test` extra: `pytest>=7`, `onnx>=1.16`.
- Build a 12-node, 512×512 MatMul model in a fixture (`pytest.importorskip("onnx")`).
- Run each scenario in a subprocess with a timeout; children call `signal.alarm(60)` and exit with `sys.exit(0)`, not `os._exit`.

| Test | Scenario | Assert |
| --- | --- | --- |
| `test_fork_preloaded_parent_child_rebuilds` | Parent runs; child rebuilds and runs | Child exit 0, output equals parent |
| `test_fork_preloaded_parent_idle_child` | Child does nothing | Child exit 0 |
| `test_fork_with_lock_held` | Parent holds `_session_lock` during fork | Child exit 0 |
| `test_fork_lazy_parent` | Parent never builds a session | Child exit 0 |
| `test_threads_share_one_session` | 16 threads call `get_session(...).run` | All outputs equal, `len(_session_cache) == 1` |
| `test_intra_op_thread_count` | Linux only: count `/proc/self/task` before and after | Exactly 3 more threads |

- Prove the tests detect the bug: temporarily disable `os.register_at_fork` and confirm `test_fork_preloaded_parent_idle_child` fails with a signal. Do not commit that change.
- Run `pytest tests/integration -q` five times and commit.

### Task 7: Documentation

**Files:** Modify `README.md`.

- Add a "Concurrency and deployment" section with the spec's mode table and notes, a Celery `worker_process_init` example using `SessionConfig.from_env(...)` and `dia.preload()`, and a plain statement that injected sessions must be created after fork.
- Add `pytest tests/integration` and the `test` extra to the development commands. Do not change accuracy or speed claims.

### Task 8: Maintainer Validation

These need a Hugging Face token, torch, or production hosts, so they can't run in a restricted environment.

1. `pytest tests/e2e -v -s` (DER 0.1487, 2 speakers) and `python scripts/benchmark.py tests/goldens/clip.wav` (CPU speed within normal variation of `docs/benchmark_results.md`).
2. GPU host, Celery prefork and threads: thread count per worker vs budget, CUDA placement, `docker stop` exit 0, shutdown/requeue hooks, and a deliberate parent preload to answer the CUDA open question.
3. Migrate a downstream service on a branch: replace its session factory with `SessionConfig`, keep its embedding thread setting, confirm VAD and diarization still share one segmentation session per worker.
4. Record results in the spec and close its open questions.

## Rollout Order

1. Tasks 1–3: additive, with no public behaviour change.
2. Tasks 4–5: the behaviour-changing tasks. Review these most closely.
3. Task 6: must pass before merging.
4. Task 7: documentation.
5. Task 8: required before tagging a release or migrating downstream services.

## Risks

| Risk | Mitigation |
| --- | --- |
| Deliberate `Py_IncRef` leak is unusual | Only runs for sessions wrongly inherited; covered by Task 6 with a failure proof |
| Errors move from constructor to first call | Documented; `preload()` at worker start fails fast |
| Fork behaviour differs by ONNX Runtime version | `onnxruntime>=1.16` is broad and only 1.31.1 was probed; run Task 6 on the oldest supported version and raise the floor if it fails |
| CUDA behaves differently after fork | Task 8, Step 2; the supported path never preloads in the parent |
| Benchmark or tests rely on private attributes | `_seg` and `_emb` are preserved; caches and `_client` are updated in this plan |
