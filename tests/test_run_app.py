import os
import sys

import run_app


def test_prepare_numba_cache_noop_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    path_before = list(sys.path)

    assert run_app.prepare_numba_cache() is None
    assert sys.path == path_before
    assert "NUMBA_CACHE_DIR" not in os.environ


def test_prepare_numba_cache_frozen_stable_source_and_cache_dir(monkeypatch, tmp_path):
    # Frozen launch: _MEIPASS is the (per-launch) extraction dir; the
    # kernels module must be copied to a stable per-user dir, that dir put
    # first on sys.path, and the cache dir pinned - numba keys its disk
    # cache by the source directory path, so the extraction dir defeats it.
    repo_dir = os.path.dirname(os.path.abspath(run_app.__file__))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", repo_dir, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    stable_src = run_app.prepare_numba_cache()

    expected_dir = os.path.join(str(tmp_path), "Roaming", "BriCoS", "numba_src")
    assert stable_src == expected_dir
    assert sys.path[0] == expected_dir
    assert os.path.isfile(os.path.join(expected_dir, "bricos_kernels.py"))
    assert os.environ["NUMBA_CACHE_DIR"] == os.path.join(
        str(tmp_path), "Roaming", "BriCoS", "numba_cache")

    # Re-running (next launch) overwrites the copy and must not fail.
    assert run_app.prepare_numba_cache() == expected_dir


def test_prepare_numba_cache_failure_never_blocks_startup(monkeypatch):
    # Startup must survive a broken environment: caching is an optimization.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", r"Z:\does\not\exist", raising=False)
    monkeypatch.setenv("APPDATA", r"Z:\also\not\writable")
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert run_app.prepare_numba_cache() is None
