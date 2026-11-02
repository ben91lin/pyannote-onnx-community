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

- [ ] **Step 1: Write failing tests**

```python
import pytest

from pyannote_onnx_community.config import SessionConfig


def test_defaults():
    c = SessionConfig()
    assert c.intra_op_threads == 0
    assert c.ort_providers() == ["CPUExecutionProvider"]


def test_provider_options_are_hashable_and_round_trip():
    opts = {"device_id": 1, "arena_extend_strategy": "kSameAsRequested"}
    c = SessionConfig(providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])
    assert hash(c) == hash(SessionConfig(providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"]))
    assert c.ort_providers() == [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]


def test_option_order_does_not_change_identity():
    a = SessionConfig(providers=[("P", {"a": 1, "b": 2})])
    b = SessionConfig(providers=[("P", {"b": 2, "a": 1})])
    assert a == b


def test_freezing_is_idempotent():
    a = SessionConfig(providers=[("P", {"a": 1})])
    assert SessionConfig(providers=a.providers) == a


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_invalid_threads(bad):
    with pytest.raises(ValueError):
        SessionConfig(intra_op_threads=bad)


def test_empty_providers_rejected():
    with pytest.raises(ValueError):
        SessionConfig(providers=())


def test_from_env(monkeypatch):
    monkeypatch.setenv("X_INTRA_OP_THREADS", "4")
    assert SessionConfig.from_env("X").intra_op_threads == 4
    monkeypatch.setenv("X_INTRA_OP_THREADS", " ")
    assert SessionConfig.from_env("X").intra_op_threads == 0
    monkeypatch.delenv("X_INTRA_OP_THREADS")
    assert SessionConfig.from_env("X", providers=["P"]).ort_providers() == ["P"]


def test_from_env_invalid(monkeypatch):
    monkeypatch.setenv("X_INTRA_OP_THREADS", "many")
    with pytest.raises(ValueError, match="X_INTRA_OP_THREADS"):
        SessionConfig.from_env("X")
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/unit/test_session_config.py -q
```

- [ ] **Step 3: Implement**

```python
import os
from dataclasses import dataclass


def _freeze_provider(provider):
    if isinstance(provider, str):
        return provider
    name, options = provider
    return (name, tuple(sorted(dict(options).items())))


@dataclass(frozen=True)
class SessionConfig:
    """ONNX Runtime session settings. Hashable, so it can key the session cache."""

    providers: tuple = ("CPUExecutionProvider",)
    intra_op_threads: int = 0  # 0 = ONNX Runtime default

    def __post_init__(self):
        if isinstance(self.providers, str):
            raise ValueError("providers must be a sequence, not a single string.")
        frozen = tuple(_freeze_provider(p) for p in self.providers)
        if not frozen:
            raise ValueError("providers must not be empty.")
        object.__setattr__(self, "providers", frozen)
        threads = self.intra_op_threads
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
            raise ValueError(f"intra_op_threads must be an integer >= 0, got {threads!r}.")

    def ort_providers(self) -> list:
        return [p if isinstance(p, str) else (p[0], dict(p[1])) for p in self.providers]

    @classmethod
    def from_env(cls, prefix: str, *, providers=None) -> "SessionConfig":
        name = f"{prefix}_INTRA_OP_THREADS"
        raw = os.environ.get(name, "").strip()
        try:
            threads = int(raw) if raw else 0
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
        kwargs = {"intra_op_threads": threads}
        if providers:
            kwargs["providers"] = tuple(providers)
        return cls(**kwargs)
```

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/unit/test_session_config.py tests/unit/test_config.py -q
```

---

### Task 2: Unified Session Cache

**Files:** Modify `_lib.py`, `tests/unit/test_lib_providers.py`; create `tests/unit/test_session_cache.py`.

- [ ] **Step 1: Write failing tests**

```python
import pytest

import pyannote_onnx_community._lib as lib
from pyannote_onnx_community.config import SessionConfig


class _FakeSession:
    created = []

    def __init__(self, model_path, sess_options=None, providers=None):
        self.options = sess_options
        self.providers = providers
        _FakeSession.created.append(self)


@pytest.fixture(autouse=True)
def fake_ort(monkeypatch):
    _FakeSession.created = []
    monkeypatch.setattr(lib.onnxruntime, "InferenceSession", _FakeSession)
    lib.clear_session_cache()
    yield
    lib.clear_session_cache()


def test_equal_config_shares_session():
    a = lib.get_session("m", SessionConfig(intra_op_threads=2))
    b = lib.get_session("m", SessionConfig(intra_op_threads=2))
    assert a is b
    assert len(_FakeSession.created) == 1


def test_different_threads_or_providers_do_not_share():
    a = lib.get_session("m", SessionConfig(intra_op_threads=1))
    b = lib.get_session("m", SessionConfig(intra_op_threads=2))
    c = lib.get_session("m", SessionConfig(providers=["P"]))
    assert len({id(a), id(b), id(c)}) == 3


def test_threads_reach_session_options():
    s = lib.get_session("m", SessionConfig(intra_op_threads=3))
    assert s.options.intra_op_num_threads == 3


def test_provider_options_reach_session():
    opts = {"device_id": 1}
    s = lib.get_session("m", SessionConfig(providers=[("P", opts)]))
    assert s.providers == [("P", opts)]


def test_legacy_loaders_accept_provider_options():
    s = lib.load_segmentation_session("m", providers=[("P", {"device_id": 1})])
    assert s.providers == [("P", {"device_id": 1})]


def test_clear_session_cache():
    a = lib.get_session("m")
    lib.clear_session_cache()
    assert lib.get_session("m") is not a


def test_after_fork_hook_leaks_and_clears(monkeypatch):
    incref = []
    monkeypatch.setattr(lib, "_leak_reference", incref.append)
    s = lib.get_session("m")
    old_lock = lib._session_lock
    lib._after_fork_in_child()
    assert incref == [s]
    assert lib._session_cache == {}
    assert lib._session_lock is not old_lock
```

- [ ] **Step 2: Replace the caches in `_lib.py`**

Remove `_seg_cache`, `_emb_cache`, and the old `_new_session`. Add `ctypes`, `os`, and `threading` imports. Update `__all__`.

```python
_session_cache: dict[tuple[str, SessionConfig], "onnxruntime.InferenceSession"] = {}
_session_lock = threading.Lock()


def _new_session(model_path: str, config: SessionConfig):
    if onnxruntime is None:
        raise ImportError(
            "onnxruntime is required for pyannote_onnx SD provider. "
            "Install via `pip install pyannote-onnx-community`."
        )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = config.intra_op_threads
    return onnxruntime.InferenceSession(
        model_path, sess_options=options, providers=config.ort_providers()
    )


def get_session(model_path: str, config: SessionConfig | None = None):
    """Return the process-wide session for ``(model_path, config)``, creating it once."""
    config = config or SessionConfig()
    key = (model_path, config)
    with _session_lock:
        session = _session_cache.get(key)
        if session is None:
            session = _new_session(model_path, config)
            _session_cache[key] = session
        return session


def clear_session_cache() -> None:
    """Drop cached sessions. No inference may be running on them."""
    with _session_lock:
        _session_cache.clear()


def _config_from_providers(providers) -> SessionConfig:
    return SessionConfig(providers=tuple(providers)) if providers else SessionConfig()


def load_segmentation_session(model_path: str, providers=None, session_config=None):
    return get_session(model_path, session_config or _config_from_providers(providers))


def load_embedding_session(model_path: str, providers=None, session_config=None):
    return get_session(model_path, session_config or _config_from_providers(providers))
```

- [ ] **Step 3: Add the fork handler**

```python
def _leak_reference(obj) -> None:
    ctypes.pythonapi.Py_IncRef(ctypes.py_object(obj))


def _after_fork_in_child() -> None:
    """Make inherited sessions harmless in a forked child.

    An inherited session's thread pool does not exist here, and destroying it
    crashes or hangs the child, even at normal interpreter exit. Keep those
    sessions alive forever and let the child build its own on demand.
    """
    global _session_lock
    _session_lock = threading.Lock()
    for session in _session_cache.values():
        _leak_reference(session)
    _session_cache.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
```

- [ ] **Step 4: Update `tests/unit/test_lib_providers.py`**

Replace `lib._seg_cache.clear()` and `lib._emb_cache.clear()` with `lib.clear_session_cache()`.

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/unit/test_session_cache.py tests/unit/test_lib_providers.py -q
```

---

### Task 3: Session Handle

**Files:** Modify `_lib.py`; add to `tests/unit/test_session_cache.py`.

- [ ] **Step 1: Write failing tests**

```python
def test_handle_with_injected_session_never_resolves(monkeypatch):
    monkeypatch.setattr(lib, "resolve_onnx_path", lambda *a: pytest.fail("resolved"))
    injected = object()
    handle = lib.make_session_handle("repo", session=injected, session_config=None, providers=None)
    assert handle.get() is injected


def test_handle_rejects_session_and_config():
    with pytest.raises(ValueError):
        lib.make_session_handle("repo", session=object(), session_config=SessionConfig(), providers=None)


def test_handle_resolves_now_but_builds_later(monkeypatch):
    monkeypatch.setattr(lib, "resolve_onnx_path", lambda *a: "m")
    handle = lib.make_session_handle("repo", session=None, session_config=None, providers=["P"])
    assert _FakeSession.created == []
    assert handle.get().providers == ["P"]


def test_handle_config_beats_providers(monkeypatch):
    monkeypatch.setattr(lib, "resolve_onnx_path", lambda *a: "m")
    handle = lib.make_session_handle("repo", session=None, session_config=SessionConfig(providers=["C"]), providers=["P"])
    assert handle.get().providers == ["C"]
```

- [ ] **Step 2: Implement**

```python
@dataclass(frozen=True, slots=True)
class SessionHandle:
    """Where to get a session. Holds no library-created session, so it survives fork."""

    path: str | None = None
    config: SessionConfig | None = None
    session: object | None = None

    def get(self):
        return self.session if self.session is not None else get_session(self.path, self.config)


def make_session_handle(repo_or_path: str, *, session, session_config, providers) -> SessionHandle:
    if session is not None:
        if session_config is not None:
            raise ValueError("Pass either an injected session or a session config, not both.")
        return SessionHandle(session=session)
    config = session_config or _config_from_providers(providers)
    return SessionHandle(path=resolve_onnx_path(repo_or_path, "onnx/model.onnx"), config=config)
```

- [ ] **Step 3: Run tests and commit**

---

### Task 4: Shared PLDA

**Files:** Modify `_plda.py`, `_pipeline.py`, `tests/unit/test_diarization.py`.

- [ ] **Step 1: Add `load_plda` to `_plda.py`**

```python
_plda_cache: dict[str, "PLDA"] = {}
_plda_lock = threading.Lock()


def load_plda(repo_id: str) -> "PLDA | None":
    """Load PLDA once per process. Read-only, so threads and forks may share it."""
    with _plda_lock:
        plda = _plda_cache.get(repo_id)
        if plda is None:
            plda = PLDA.from_pretrained(repo_id, subfolder="plda")
            if plda is not None:
                _plda_cache[repo_id] = plda
        return plda


def _reset_lock_after_fork() -> None:
    global _plda_lock
    _plda_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)
```

- [ ] **Step 2: Use it in `PyannoteOnnxClient._load_plda`**

```python
from pyannote_onnx_community._plda import load_plda

self.plda = load_plda(self.config.plda_repo_id)
```

Keep the existing `RuntimeError` when the result is `None`. Keep the lazy import so empty audio never loads PLDA.

- [ ] **Step 3: Stub PLDA in `tests/unit/test_diarization.py`**

Add to `_make_dia`:

```python
monkeypatch.setattr("pyannote_onnx_community._plda.load_plda", lambda repo_id: object())
```

The clustering stub ignores the object. This removes the network dependency behind the 7 baseline failures.

- [ ] **Step 4: Add a cache test**

Monkeypatch `PLDA.from_pretrained` to count calls. Assert that two `load_plda` calls load once, and that a `None` result is not cached. Clear `_plda_cache` in a fixture.

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/unit/test_diarization.py tests/unit/test_pipeline.py -q
```

---

### Task 5: Lazy Public Classes

**Files:** Modify `diarization.py`, `vad.py`, `embedding.py`, `__init__.py`; create `tests/unit/test_lazy_sessions.py`.

- [ ] **Step 1: Write failing tests**

Monkeypatch `lib.resolve_onnx_path` to return a fake path and `lib.onnxruntime.InferenceSession` to a counting fake, then assert for all three classes:

- The constructor creates no session.
- `preload()` creates the expected sessions, and a second call creates none. `ONNXSpeakerDiarization.preload()` also calls `load_plda`, stubbed here.
- VAD and diarization with equal configs reuse one segmentation session.
- `providers=["P"]` and `seg_session_config=` reach the session as specified.
- Injected session plus config raises `ValueError`.
- `dia._seg` returns the live session.

- [ ] **Step 2: Update `ONNXSpeakerDiarization`**

```python
def __init__(
    self,
    segmentation_repo: str = SEGMENTATION_REPO,
    embedding_repo: str = EMBEDDING_REPO,
    config: SDConfig | None = None,
    seg_session=None,
    emb_session=None,
    providers: list | None = None,
    seg_session_config: SessionConfig | None = None,
    emb_session_config: SessionConfig | None = None,
):
    self.config = config or SDConfig()
    self._seg_handle = make_session_handle(
        segmentation_repo, session=seg_session, session_config=seg_session_config, providers=providers
    )
    self._emb_handle = make_session_handle(
        embedding_repo, session=emb_session, session_config=emb_session_config, providers=providers
    )

@property
def _seg(self):
    return self._seg_handle.get()

@property
def _emb(self):
    return self._emb_handle.get()

def preload(self) -> None:
    """Build sessions and load PLDA now. Call after fork, in the process that infers."""
    self._seg, self._emb
    PyannoteOnnxClient(seg_session=None, emb_session=None, config=self.config)._load_plda()
```

In `__call__`, build the client per call:

```python
client = PyannoteOnnxClient(seg_session=self._seg, emb_session=self._emb, config=self.config)
result = client(audio_input=wav, ...)
```

Keep new arguments after the existing ones so positional callers are unaffected.

- [ ] **Step 3: Update `ONNXVoiceActivityDetection` and `ONNXSpeakerEmbedding` the same way**

`__call__` and `preload()` read the `_seg` or `_emb` property. Embedding has no PLDA.

- [ ] **Step 4: Export the new API**

Add `SessionConfig` and `clear_session_cache` to `__init__.py` and `__all__`.

- [ ] **Step 5: Run the full suite and commit**

```bash
pytest tests/unit tests/parity -q
```

Expected: no failures. Parity tests that need models may still skip. `scripts/benchmark.py` needs no change because `_seg` is kept.

---

### Task 6: Real-Runtime Concurrency Tests

**Files:** Modify `pyproject.toml`; create `tests/integration/__init__.py` and `tests/integration/test_concurrency.py`.

- [ ] **Step 1: Add a torch-free test extra**

```toml
test = [
    "pytest>=7",
    "onnx>=1.16",
]
```

- [ ] **Step 2: Build a tiny model fixture**

`pytest.importorskip("onnx")`, then write a 12-node, 512×512 MatMul model to `tmp_path_factory`. The chain must be large enough to use the pool.

- [ ] **Step 3: Run each scenario in a subprocess**

Fork inside the pytest process would inherit pytest's state. Use this helper with a watchdog, because the failure modes are hangs and signals:

```python
def _run(script: str, model: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script, model],
        capture_output=True, text=True, timeout=120,
    )
```

Each script builds a session through `lib.get_session(path, SessionConfig(intra_op_threads=4))`, forks where relevant, and has the child call `signal.alarm(60)` and exit with `sys.exit(0)`, not `os._exit`. Normal exit is the path that crashes without the fork handler. Skip fork scenarios on platforms without `os.fork`.

| Test | Scenario | Assert |
| --- | --- | --- |
| `test_fork_preloaded_parent_child_rebuilds` | Parent runs; child rebuilds and runs | Child exit 0, output equals parent |
| `test_fork_preloaded_parent_idle_child` | Child does nothing | Child exit 0 |
| `test_fork_with_lock_held` | Parent holds `_session_lock` during fork | Child exit 0 |
| `test_fork_lazy_parent` | Parent never builds a session | Child exit 0 |
| `test_threads_share_one_session` | 16 threads call `get_session(...).run` | All outputs equal, `len(_session_cache) == 1` |
| `test_intra_op_thread_count` | Linux only: count `/proc/self/task` before and after | Exactly 3 more threads |

- [ ] **Step 4: Prove the tests detect the bug**

Temporarily disable `os.register_at_fork` and confirm `test_fork_preloaded_parent_idle_child` fails with a signal. Restore it. Do not commit the temporary change.

- [ ] **Step 5: Run several times and commit**

```bash
for i in 1 2 3 4 5; do pytest tests/integration -q || exit 1; done
```

---

### Task 7: Documentation

**Files:** Modify `README.md`.

- [ ] **Step 1: Add a "Concurrency and deployment" section**

Include the mode table and notes from the spec, plus:

```python
import os
from celery.signals import worker_process_init  # threads pool: worker_init

from pyannote_onnx_community import ONNXSpeakerDiarization, SessionConfig

dia = ONNXSpeakerDiarization(
    seg_session_config=SessionConfig.from_env("PYANNOTE_ONNX_SEGMENTATION"),
    emb_session_config=SessionConfig.from_env("PYANNOTE_ONNX_EMBEDDING"),
)


@worker_process_init.connect
def init_models(**_):
    dia.preload()
```

State plainly that injected sessions must be created after fork.

- [ ] **Step 2: Update development commands**

Add `pytest tests/integration` and the `test` extra. Do not change any accuracy or speed claims in this work.

- [ ] **Step 3: Commit**

---

### Task 8: Maintainer Validation

These need a Hugging Face token, torch, or production hosts, so they can't run in a restricted environment.

- [ ] **Step 1: Output and speed regression, dev venv**

```bash
pytest tests/e2e -v -s
python scripts/benchmark.py tests/goldens/clip.wav
```

Expected: DER 0.1487 with 2 speakers, and CPU speed within normal variation of `docs/benchmark_results.md`.

- [ ] **Step 2: GPU host, prefork and threads**

Set `PYANNOTE_ONNX_SEGMENTATION_INTRA_OP_THREADS` to the host's per-worker budget, then run real tasks.

Check:

- Thread count per worker, compared with the budget.
- CUDA session placement on each worker's GPU.
- Worker exit status from `docker stop`, which should be 0 and not 137 or a signal.
- Any application-level shutdown or requeue hook still working.
- Deliberately preloading in the parent, to answer the spec's CUDA open question.

- [ ] **Step 3: Migrate a downstream service on a branch**

Replace its session factory with `SessionConfig`, keeping its current embedding thread setting. Confirm that VAD and diarization still share one segmentation session per worker.

- [ ] **Step 4: Record results**

Add the findings to the spec, and close its open questions.

---

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
