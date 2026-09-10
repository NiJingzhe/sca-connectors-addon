"""Executable spec for the FEMaster solver wrapper.

Tests use a fake solver script — the real binary is only needed for the
end-to-end suite. The wrapper owns: binary location, subprocess execution
with timeout, exit-code handling, and locating the .frd/.res outputs.
"""

import os
import stat
import textwrap

import pytest

from connverify.solver import (
    SolveResult,
    SolverError,
    SolverNotFoundError,
    locate_femaster,
    run_solve,
)

FAKE_OK = """\
#!/usr/bin/env python3
import sys, pathlib
deck = sys.argv[sys.argv.index("--deck")] if "--deck" in sys.argv else sys.argv[1]
stem = pathlib.Path(deck).with_suffix("")
stem.with_suffix(".frd").write_text("fake frd\\n")
stem.with_suffix(".res").write_text("fake res\\n")
print("[INFO] fake solve done")
"""

FAKE_FAIL = """\
#!/usr/bin/env python3
import sys
sys.stderr.write("[ERROR] parse error: unknown keyword *BOGUS\\n")
sys.exit(3)
"""

FAKE_HANG = """\
#!/usr/bin/env python3
import time
time.sleep(30)
"""


@pytest.fixture
def fake_bin(tmp_path):
    def _write(body: str, name: str = "FakeFEMaster") -> str:
        p = tmp_path / name
        p.write_text(textwrap.dedent(body))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return str(p)
    return _write


@pytest.fixture
def deck_file(tmp_path):
    p = tmp_path / "case_op.inp"
    p.write_text("*MODEL, NAME=T\n*END\n")
    return str(p)


class TestLocateFemaster:
    def test_explicit_path_wins(self, fake_bin):
        assert locate_femaster(explicit=fake_bin(FAKE_OK)) == fake_bin(FAKE_OK)

    def test_env_var_is_honoured(self, fake_bin, monkeypatch):
        path = fake_bin(FAKE_OK)
        monkeypatch.setenv("CONNVERIFY_FEMASTER", path)
        assert locate_femaster() == path

    def test_missing_binary_names_every_candidate(self, monkeypatch):
        from connverify import solver as pr
        monkeypatch.delenv("CONNVERIFY_FEMASTER", raising=False)
        monkeypatch.delenv("FEMASTER", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr(pr, "_vendor_candidate", lambda: None)
        with pytest.raises(SolverNotFoundError) as ei:
            locate_femaster()
        msg = str(ei.value)
        assert "CONNVERIFY_FEMASTER" in msg
        assert "vendor" in msg or "release" in msg


class TestRunSolve:
    def test_successful_solve_reports_outputs(self, fake_bin, deck_file):
        result = run_solve(deck_file, binary=fake_bin(FAKE_OK))
        assert isinstance(result, SolveResult)
        assert result.returncode == 0
        assert result.frd_path and os.path.exists(result.frd_path)
        assert result.res_path and os.path.exists(result.res_path)
        assert "fake solve done" in result.stdout

    def test_nonzero_exit_raises_with_stderr_tail(self, fake_bin, deck_file):
        with pytest.raises(SolverError) as ei:
            run_solve(deck_file, binary=fake_bin(FAKE_FAIL))
        assert "parse error" in str(ei.value)
        assert "*BOGUS" in str(ei.value)

    def test_timeout_raises_and_names_the_timeout(self, fake_bin, deck_file):
        with pytest.raises(SolverError) as ei:
            run_solve(deck_file, binary=fake_bin(FAKE_HANG), timeout_s=0.5)
        assert "timeout" in str(ei.value).lower()

    def test_missing_outputs_are_reported_not_crashed(self, fake_bin, deck_file):
        no_out = FAKE_OK.replace('stem.with_suffix(".frd")', 'x = 1 #').replace(
            'stem.with_suffix(".res")', 'x = 1 #')
        result = run_solve(deck_file, binary=fake_bin(no_out))
        assert result.frd_path is None
        assert result.res_path is None

    def test_extra_args_are_forwarded(self, fake_bin, deck_file):
        seen = """\
        #!/usr/bin/env python3
        import sys, pathlib
        pathlib.Path(sys.argv[-1]).parent.joinpath("args.txt").write_text(" ".join(sys.argv))
        deck = sys.argv[1]
        stem = pathlib.Path(deck).with_suffix("")
        stem.with_suffix(".frd").write_text("x")
        stem.with_suffix(".res").write_text("x")
        """
        result = run_solve(deck_file, binary=fake_bin(seen), extra_args=["--ncpus", "4"])
        args = open(os.path.join(os.path.dirname(result.deck_path), "args.txt")).read()
        assert "--ncpus" in args and "4" in args
