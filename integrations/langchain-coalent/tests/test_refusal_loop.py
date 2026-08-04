"""LG4 — the refusal-loop example runs to completion with a key confirmed.

The example is executed exactly as a user would run it (a subprocess of the same
interpreter), so this also proves it is runnable stand-alone and offline."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "refusal_loop.py"


def test_lg4_refusal_loop_example_runs_to_completion() -> None:
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)          # the example must be offline-complete
    env.pop("ANTHROPIC_API_KEY", None)
    proc = subprocess.run(
        [sys.executable, str(_EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"example failed:\n{proc.stdout}\n{proc.stderr}"
    assert "REFUSE" in proc.stdout            # the conditional edge actually fired
    assert "45 days" in proc.stdout           # the lost fact was recovered verbatim
    assert "key_confirmed" in proc.stdout     # the confirmed retry attached the key
    assert "refusal loop complete" in proc.stdout
