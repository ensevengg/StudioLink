"""Adversarial checks at filesystem and process boundaries."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from studiolink.state import StateStore


def test_interrupted_atomic_state_write_preserves_old_json(tmp_path, monkeypatch):
    from conftest import make_sync_record

    state = StateStore(tmp_path / "state.json")
    state.upsert(make_sync_record("old:1"))
    old = state.path.read_bytes()

    def failed_replace(*args, **kwargs):
        raise OSError(errno.ENOSPC, "disk full")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failed_replace)
        with pytest.raises(OSError):
            state.upsert(make_sync_record("new:1"))
    assert state.path.read_bytes() == old
    assert set(state.get_all_records()) == {"old:1"}
    state.upsert(make_sync_record("new:1"))
    assert set(state.get_all_records()) == {"old:1", "new:1"}


def test_os_releases_lock_after_killed_process(tmp_path):
    state = StateStore(tmp_path / "state.json")
    code = """
import sys, time
from pathlib import Path
from studiolink.state import StateStore
with StateStore(Path(sys.argv[1])).lock():
    print("LOCKED", flush=True)
    time.sleep(30)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(state.path)],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="Another StudioLink"):
            with state.lock():
                pytest.fail("overlapping transaction was allowed")
    finally:
        child.kill()
        child.wait(timeout=5)
    with state.lock():
        assert True
