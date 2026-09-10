"""FEMaster solver wrapper: locate the binary, run a deck, collect outputs.

Binary resolution order:
1. explicit ``binary=`` argument
2. ``$CONNVERIFY_FEMASTER``
3. ``$FEMASTER``
4. ``FEMaster`` on ``PATH``
5. a dev checkout's ``vendor/FEMaster``

A missing binary is a loud stop — never a silent analysis skip.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class SolverError(RuntimeError):
    """The solver run failed (non-zero exit, timeout, or crash)."""


class SolverNotFoundError(RuntimeError):
    """No usable FEMaster binary could be located."""


@dataclass(frozen=True)
class SolveResult:
    deck_path: str
    returncode: int
    stdout: str
    stderr: str
    frd_path: Optional[str]
    res_path: Optional[str]
    duration_s: float


def _vendor_candidate() -> Optional[str]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "vendor" / "FEMaster"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def locate_femaster(explicit: Optional[str] = None) -> str:
    candidates: List[Optional[str]] = [
        explicit,
        os.environ.get("CONNVERIFY_FEMASTER"),
        os.environ.get("FEMASTER"),
        shutil.which("FEMaster") or shutil.which("femaster"),
        _vendor_candidate(),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SolverNotFoundError(
        "FEMaster binary not found. Tried: binary= argument, "
        "$CONNVERIFY_FEMASTER, $FEMASTER, PATH, <repo>/vendor/FEMaster. "
        "Install it from https://github.com/Luecx/FEMaster/releases "
        "(or run tools/fetch_femaster.sh in a dev checkout) and point "
        "CONNVERIFY_FEMASTER at the executable."
    )


def run_solve(
    deck_path: str,
    *,
    binary: Optional[str] = None,
    timeout_s: float = 600.0,
    extra_args: Optional[List[str]] = None,
    ncpus: Optional[int] = None,
) -> SolveResult:
    exe = locate_femaster(explicit=binary)
    deck = Path(deck_path)
    if not deck.is_file():
        raise SolverError(f"deck not found: {deck}")

    cmd = [exe, str(deck)]
    if ncpus:
        cmd += ["--ncpus", str(ncpus)]
    if extra_args:
        cmd += list(extra_args)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=str(deck.parent), capture_output=True, text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise SolverError(
            f"solver exceeded timeout of {timeout_s} s for deck {deck.name}; "
            "reduce mesh size or raise timeout_s"
        ) from exc
    duration = time.monotonic() - started

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise SolverError(
            f"FEMaster exited with code {proc.returncode} on {deck.name}.\n"
            f"stderr tail:\n{tail}"
        )

    frd = deck.with_suffix(".frd")
    res = deck.with_suffix(".res")
    return SolveResult(
        deck_path=str(deck),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        frd_path=str(frd) if frd.is_file() else None,
        res_path=str(res) if res.is_file() else None,
        duration_s=duration,
    )
