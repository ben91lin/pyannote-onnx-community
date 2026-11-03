"""Concurrency and fork behaviour against a real (tiny) ONNX Runtime session.

Each scenario runs in a fresh interpreter: forking inside pytest would inherit
pytest's own state, and the failure modes under test are hangs and signals.

Forked children end with ``os._exit``, as billiard and multiprocessing do.
A child that ends with ``exit()`` hangs in ONNX Runtime's own C++ static
destructors whenever the parent imported onnxruntime, with or without any
session, so that path is outside the library's reach. To exercise the
destructor path the fork handler guards, preloaded-parent children explicitly
drop the cache and collect garbage before exiting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")

_DIM = 512
_NODES = 12


@pytest.fixture(scope="module")
def model(tmp_path_factory) -> str:
    """A chain of 512x512 MatMuls: big enough for ORT to use its thread pool."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    nodes, inits = [], []
    prev = "x"
    for i in range(_NODES):
        w = (rng.standard_normal((_DIM, _DIM)) / np.sqrt(_DIM)).astype(np.float32)
        inits.append(numpy_helper.from_array(w, name=f"w{i}"))
        out = "y" if i == _NODES - 1 else f"h{i}"
        nodes.append(helper.make_node("MatMul", [prev, f"w{i}"], [out]))
        prev = out
    graph = helper.make_graph(
        nodes,
        "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [_DIM, _DIM])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [_DIM, _DIM])],
        inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    path = tmp_path_factory.mktemp("model") / "chain.onnx"
    onnx.save(m, str(path))
    return str(path)


_PRELUDE = textwrap.dedent(
    """
    import gc, os, signal, sys, threading
    import numpy as np
    import pyannote_onnx_community._lib as lib
    from pyannote_onnx_community.config import SessionConfig

    MODEL = sys.argv[1]
    CFG = SessionConfig(intra_op_threads=4)
    X = np.random.default_rng(1).standard_normal((512, 512)).astype(np.float32)

    def infer():
        return lib.get_session(MODEL, CFG).run(None, {"x": X})[0]

    def release_sessions():
        # Drop every cache reference and collect: without the fork handler this
        # destroys a session inherited from the parent, which crashes.
        lib.clear_session_cache()
        gc.collect()

    def fork_child(body):
        pid = os.fork()
        if pid == 0:
            signal.alarm(60)
            code = 0
            try:
                body()
                release_sessions()
            except BaseException as exc:
                print("CHILD_ERROR", repr(exc), flush=True)
                code = 4
            sys.stdout.flush()
            os._exit(code)
        _, status = os.waitpid(pid, 0)
        if os.WIFSIGNALED(status):
            print("CHILD_SIGNAL", os.WTERMSIG(status))
            sys.exit(3)
        code = os.WEXITSTATUS(status)
        print("CHILD_EXIT", code)
        sys.exit(code)
    """
)


def _run(script: str, model: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _PRELUDE + textwrap.dedent(script), model],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"


@needs_fork
def test_fork_preloaded_parent_child_rebuilds(model):
    proc = _run(
        """
        expected = infer()

        def body():
            assert lib._session_cache == {}, "inherited session still cached"
            got = infer()
            assert np.allclose(got, expected, rtol=1e-5, atol=1e-5), "child output differs"

        fork_child(body)
        """,
        model,
    )
    _assert_ok(proc)


@needs_fork
def test_fork_preloaded_parent_idle_child(model):
    proc = _run(
        """
        infer()
        fork_child(lambda: None)
        """,
        model,
    )
    _assert_ok(proc)


@needs_fork
def test_fork_with_lock_held(model):
    proc = _run(
        """
        infer()
        with lib._session_lock:
            # Fork while the cache lock is held: the child must get a fresh lock.
            fork_child(lambda: infer())
        """,
        model,
    )
    _assert_ok(proc)


@needs_fork
def test_fork_lazy_parent(model):
    proc = _run(
        """
        fork_child(lambda: infer())
        """,
        model,
    )
    _assert_ok(proc)


def test_threads_share_one_session(model):
    proc = _run(
        """
        results = [None] * 16
        barrier = threading.Barrier(16)

        def work(i):
            barrier.wait()
            results[i] = infer()

        threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(np.array_equal(r, results[0]) for r in results), "outputs differ"
        assert len(lib._session_cache) == 1, len(lib._session_cache)
        """,
        model,
    )
    _assert_ok(proc)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="counts /proc/self/task")
def test_intra_op_thread_count(model):
    # "+3 for intra_op_threads=4" is a property of the ONNX Runtime build, not of
    # this library; it held on the 1.20.1, 1.29.0 and 1.31.1 pip wheels. If a
    # build ever differs, loosen this to ">= 1 and <= 3" instead of skipping it:
    # the point is that the setting reaches the session and caps the pool.
    proc = _run(
        """
        before = len(os.listdir("/proc/self/task"))
        infer()
        after = len(os.listdir("/proc/self/task"))
        print("DELTA", after - before)
        assert after - before == 3, (before, after)
        """,
        model,
    )
    _assert_ok(proc)
