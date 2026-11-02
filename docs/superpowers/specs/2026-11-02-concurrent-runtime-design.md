# Concurrent Runtime Design

Status: implemented (plan Tasks 1–7). Maintainer validation on real hosts (Task 8) is pending.

## Goal

Make `pyannote-onnx-community` safe and predictable under Celery prefork, Celery threads, and plain multi-threaded services, without forcing applications to build their own ONNX Runtime sessions.

Existing `providers=` and `seg_session=` / `emb_session=` usage must keep working, and diarization output must not change.

## Problems To Solve

1. **CPU oversubscription.** `_lib._new_session` uses default `SessionOptions`, so `intra_op_num_threads=0` sizes the pool from the host. On a large multi-worker host, every worker process can open a pool sized for the whole machine, so N workers on C cores can mean up to N × C threads competing for the same cores.
2. **No thread setting in the API.** Applications must bypass the library to set a limit, and then have to recreate the session sharing they lost by doing so.
3. **Sessions are created in constructors.** A module-level `ONNXSpeakerDiarization()` in a Celery task module creates sessions in the parent before it forks.
4. **Provider options crash.** The cache key hashes `providers`, so `[("CUDAExecutionProvider", {"device_id": 1})]` raises `TypeError: unhashable type: 'dict'`.
5. **Cache and PLDA loading are unsynchronised.** Concurrent first calls can build duplicate sessions, and each instance loads its own PLDA.
6. **Cached sessions can never be released.** The module-level caches have no eviction path.

## Evidence

Probed with a small MatMul model on onnxruntime 1.31.1, Python 3.13, Linux, CPU provider, 2 vCPU.

| Scenario | Result |
| --- | --- |
| `intra_op_num_threads=4` | Adds 3 threads to the process; the caller is the fourth |
| Inherited session, child calls `run()` | Correct output, but the child has 1 thread: the pool did not survive the fork |
| Inherited session, child exits normally | `SIGSEGV`, even if the child never used the session |
| Inherited session, child deletes it | `SIGSEGV` |
| Inherited session, kept in a Python list, child exits normally | Hang until the watchdog alarm fired |
| Inherited session, leaked with `Py_IncRef`, child builds a new session | Exit 0, correct output, new pool has its threads |
| No session in parent, child builds, runs, deletes, exits | Exit 0 on the original probe host; see the re-check below |
| 8 threads sharing one session | Correct output, no extra pool threads |
| 16 threads through the prototype cache | Correct output, no duplicate session |
| Prototype of this design, parent preloaded or lock held at fork | Exit 0 across 3 repeated runs, child rebuilt with 4 threads |

Conclusions:

- Creating sessions before fork is the failure to design out. The danger is mainly at teardown, not during `run()`, so it can look healthy until a worker exits.
- Clearing the cache in the child is not enough, because dropping the last reference runs the destructor and crashes. Inherited sessions must be leaked on purpose.
- Python 3.12+ also emits a `DeprecationWarning` when a process with live pool threads forks.

Re-check during implementation (2026-11-02, Linux container, 4 vCPU, Python 3.13, onnxruntime 1.29.0 and 1.31.1):

| Scenario | Result |
| --- | --- |
| Parent only imported onnxruntime, child exits via `exit()` / `sys.exit()` | Hang. gdb: `__run_exit_handlers` → ORT static destructor → `pthread_cond_destroy`, waiting on a condition variable used by a thread ORT starts at import. Same with `disable_telemetry_events()` and with `OMP_NUM_THREADS=1` |
| Same, child exits via `os._exit()` | Exit 0 |
| Parent built a session, child drops it (`del` + `gc.collect()`), `os._exit()` | `SIGSEGV` |
| Parent built a session, child keeps it, builds its own, `os._exit()` | Exit 0 |

- The `exit()` hang happens with or without sessions, so it is outside the library's reach. The library must not try to work around it.
- billiard and `multiprocessing` end forked children with `os._exit`, so Celery prefork is not affected. Code that calls `os.fork()` by hand must end the child with `os._exit`.
- The inherited-session destructor crash is confirmed, so the fork handler is still needed. The integration tests exercise it by dropping the cache in the child before `os._exit`.
- `tests/integration` passes on onnxruntime 1.20.1, 1.29.0 and 1.31.1. Python 3.13 has no 1.16 wheel, so the floor itself was not tested.

Not covered: CUDA, CoreML, billiard's child exit path, and machines with many cores. The acceptance plan below covers these on real hosts.

## Design

### `SessionConfig`

New public, immutable, hashable value in `config.py`:

```python
@dataclass(frozen=True)
class SessionConfig:
    providers: tuple = ("CPUExecutionProvider",)
    intra_op_threads: int = 0  # 0 = ONNX Runtime default

    def ort_providers(self) -> list: ...

    @classmethod
    def from_env(cls, prefix: str, *, providers=None) -> "SessionConfig": ...
```

- `providers` accepts names and `(name, options)` pairs. `__post_init__` freezes options into sorted tuples, so the config can itself be a cache key. `ort_providers()` converts back to what ONNX Runtime expects.
- `intra_op_threads` must be an integer `>= 0`; otherwise `ValueError`.
- An empty `providers` raises `ValueError`.
- `from_env` reads `{prefix}_INTRA_OP_THREADS`. Unset or blank means `0`; invalid values raise `ValueError` so a bad deployment fails at startup.
- The library never reads the environment implicitly. Applications opt in, which keeps library behaviour visible from the call site.

Standard names:

- `PYANNOTE_ONNX_SEGMENTATION_INTRA_OP_THREADS`
- `PYANNOTE_ONNX_EMBEDDING_INTRA_OP_THREADS`

### Constructor API

Added as keyword arguments:

| Class | New arguments | New method |
| --- | --- | --- |
| `ONNXSpeakerDiarization` | `seg_session_config`, `emb_session_config` | `preload()` |
| `ONNXVoiceActivityDetection` | `seg_session_config` | `preload()` |
| `ONNXSpeakerEmbedding` | `emb_session_config` | `preload()` |

Resolution per model, highest first:

1. Injected `seg_session` / `emb_session`
2. `*_session_config`
3. `providers`
4. `CPUExecutionProvider`

Passing an injected session and a config for the same model raises `ValueError`, because one would be silently ignored. Passing `providers` with one model's config is valid: the config applies to that model and `providers` to the other.

### Lazy Sessions

- Constructors resolve and download model files, so a missing model or a gated-access failure still surfaces at construction. This also lets the parent download once before forking.
- `InferenceSession` creation happens on first inference or on `preload()`.
- Instances hold a small handle, `(path, config)` or an injected session, never a library-created session.
- Each call fetches sessions through the handle, so a forked child cannot reuse a parent's session through an instance.
- `_seg` and `_emb` remain as properties that return the live session. `scripts/benchmark.py` uses `dia._seg.get_providers()`.
- `PyannoteOnnxClient` keeps its signature and is built per call. It holds no state beyond its inputs, so concurrent calls do not share mutable data.

### Session Cache

One process-wide cache in `_lib.py`, keyed by `(model_path, SessionConfig)`:

- `get_session(path, config)` holds a lock across lookup and creation, so concurrent first calls build one session.
- VAD and diarization share a segmentation session when their configs match. Different thread counts or providers get separate sessions.
- `load_segmentation_session` and `load_embedding_session` keep working, with an added optional `session_config`.
- New public `clear_session_cache()` drops the cache's references. Callers must ensure no inference is running.

### Fork Handling

`os.register_at_fork(after_in_child=...)`, guarded by `hasattr` for platforms without fork:

1. Replace the cache lock, since another thread may have held it during the fork.
2. Call `ctypes.pythonapi.Py_IncRef` on every cached session so its destructor never runs in the child.
3. Clear the cache so the child builds its own session on demand.

The leak is deliberate and limited to memory inherited from the parent. Correct use never triggers it, because nothing is preloaded in the parent; it exists so a mistaken preload degrades to wasted memory instead of a crash at worker exit.

Injected sessions belong to the caller, so the library cannot protect them. The README will state that they must be created after fork.

### PLDA

`_plda.load_plda(repo_id)` caches per process under a lock. PLDA holds read-only NumPy arrays and pure transforms, so it may be shared between threads and inherited across fork. Only the lock is replaced in the child.

An explicitly passed `plda=` on `PyannoteOnnxClient` keeps priority. A failed load is not cached.

### Deployment Guidance

Added to the README:

| Mode | Sessions | Thread budget | Where to call `preload()` |
| --- | --- | --- | --- |
| Celery prefork | One set per child | `workers × intra ≈ allocated cores` | `worker_process_init` |
| Celery threads | One set per worker | `intra` = cores for this worker | `worker_init` |
| Multi-threaded service | One set per process | `intra` = cores for this process | Application startup |
| gevent / eventlet | Not recommended | Inference blocks the event loop | Use a separate worker |

The guidance also covers:

- Setting `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS`, because PLDA and VBx run through NumPy and SciPy.
- ONNX Runtime sizing from the host, not necessarily from a container CPU limit.
- Pinning GPUs with provider options or `CUDA_VISIBLE_DEVICES`.
- Inference being free of side effects, so requeueing a task is safe at this layer.

## Compatibility

| Behaviour | Change |
| --- | --- |
| `providers=`, `seg_session=`, `emb_session=` | Unchanged |
| Default thread count | Unchanged (`0`) |
| Diarization, VAD, and embedding output | Unchanged; no algorithm code is touched |
| Missing model or gated access | Still raises at construction |
| Corrupt model or unavailable provider | Now raises at `preload()` or first call instead of construction |
| First inference latency | Includes session creation unless `preload()` is called |
| Private `_seg_cache`, `_emb_cache`, `_client` | Removed; no external API promised them |

## Non-Goals

- Fixing the pipeline differences from upstream pyannote.audio, such as binarization, warm-up trimming, and embedding masking.
- Application shutdown handling. `exec`, warm shutdown, and requeue hooks stay in the application.
- Concurrency limits inside the library. Celery concurrency already bounds this; a semaphore can be added if measurements justify it.
- `inter_op_num_threads`, which sequential execution does not need.
- ONNX Runtime spinning controls. Revisit if idle CPU remains high after thread limits.
- Batched embedding, changing VAD defaults, and PyPI publishing.

## Acceptance Criteria

Automated:

1. `SessionConfig` validation, freezing, hashing, and `from_env` are unit tested.
2. Provider options no longer raise.
3. Constructors create no session until first use or `preload()`.
4. Sessions are shared for equal configs and separate for different ones.
5. Against a real tiny ONNX model, in subprocesses with a timeout:
   - Fork with a preloaded parent exits 0 and returns correct output.
   - An idle child of a preloaded parent exits 0.
   - Fork while the cache lock is held does not deadlock.
   - 16 threads produce correct output and build one session.
   - `intra_op_threads=4` adds 3 threads on Linux.
6. `pytest tests/unit tests/parity` passes without network access to Hugging Face, except parity tests that already skip without models.

Maintainer, on real hosts:

7. `tests/e2e` still reports DER 0.1487 on `clip.wav`.
8. CPU benchmark at default settings stays within normal run-to-run variation.
9. Celery prefork and threads workers on a GPU host process tasks, stay within the expected thread count, and stop with exit 0 on `docker stop`.
10. A downstream service with its own session factory can drop it and keep segmentation sharing and its thread settings.

## Open Questions

1. **Embedding thread default.** Embedding runs small batch-1 calls, and at least one downstream deployment pins it to 1 thread for that reason. This design keeps the library default at `0` for compatibility. Change it only if benchmarks show a clear gain across machines.
2. **CUDA after fork.** The fork handler has only been exercised on CPU. Whether a leaked inherited CUDA session is safe must be checked on a GPU host. The supported path remains not preloading in the parent.
