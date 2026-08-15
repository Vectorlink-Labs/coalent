"""Guard: the OPENAI_API_KEY-gated tier must SKIP, never crash, when the SDK is absent.

Regression: with a key exported but the openai package not installed (a plain
``pip install -e ".[dev]"`` env), the v0.4 accuracy-tier tests used to RUN and die with
ModuleNotFoundError inside the provider adapter. The tier's skip condition now requires
the SDK to be importable too; this pins that by running the tier modules in a child
pytest with a fake key. Only meaningful in an SDK-less env — with the SDK installed a
fake key would reach the network, so the guard skips itself there.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TIER = ["tests/test_v04_openai.py", "tests/test_v04_extract_gate.py"]


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is not None,
    reason="guard targets SDK-less envs; with the SDK installed the tier runs for real",
)
def test_openai_tier_skips_not_crashes_without_sdk() -> None:
    env = {**os.environ, "OPENAI_API_KEY": "sk-guard-fake-never-sent"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *_TIER, "-rs", "-p", "no:cacheprovider"],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (proc.stdout + proc.stderr).lower()
    assert proc.returncode == 0, f"tier must skip cleanly without the SDK; got:\n{out}"
    assert "failed" not in out, f"tier must skip, not fail, without the SDK:\n{out}"
    assert "skipped" in out, f"expected the tier to report skips:\n{out}"
