"""One-call orchestration: environment + package -> checks -> mesh -> solve
-> geometry-directed report.

Run directory layout (all durable artifacts):

::

    <out_dir>/
      decks/<CASE>.inp        generated FEMaster decks
      decks/<CASE>.frd/.res   solver outputs (next to their decks)
      report.json             machine contract
      report.md               agent-readable rendering
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .conncheck import check_interface
from .deck import generate_decks
from .envelope import check_envelopes
from .env import VerificationEnv
from .frd import parse_frd
from .meshing import mesh_part
from .package_reader import load_part, require_interfaces
from .report import CaseOutcome, build_report
from .solver import run_solve


def verify(
    env_source: Union[str, Path, VerificationEnv],
    *,
    out_dir: Union[str, Path] = "out/verify",
    mesh_size_mm: Optional[float] = None,
    femaster: Optional[str] = None,
    timeout_s: float = 600.0,
    ncpus: Optional[int] = None,
    render_tag_check: bool = True,
):
    """Run the full static-connector verification and write the report."""
    env, package_path = _load_env_and_package(env_source)
    env.validate()

    loaded = load_part(package_path)
    require_interfaces(loaded, env.required_interface_names())

    conn_results = tuple(check_interface(loaded, i) for i in env.interfaces)
    envelope_results = check_envelopes(loaded, env.envelopes)

    mesh = mesh_part(loaded, mesh_size_mm=mesh_size_mm)
    decks = generate_decks(mesh, env)

    out = Path(out_dir)
    deck_dir = out / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)

    render_facts = None
    if render_tag_check:
        from .render import render_tag_review

        render_facts = render_tag_review(
            mesh, out / "tag_review.png",
            title=f"tag review — {env.name}",
        )

    outcomes = []
    for case in env.load_cases:
        deck_path = deck_dir / f"{case.name}.inp"
        deck_path.write_text(decks[case.name])
        try:
            solved = run_solve(str(deck_path), binary=femaster,
                               timeout_s=timeout_s, ncpus=ncpus)
            frd = parse_frd(solved.frd_path) if solved.frd_path else None
            outcomes.append(CaseOutcome(
                name=case.name, frd=frd,
                error=None if frd is not None else
                "solver produced no .frd result file",
                duration_s=solved.duration_s,
            ))
        except Exception as exc:  # solver/deck errors become case failures
            outcomes.append(CaseOutcome(
                name=case.name, frd=None, error=str(exc), duration_s=None))

    report = build_report(
        env=env, mesh=mesh,
        conn_results=conn_results, envelope_results=envelope_results,
        outcomes=outcomes, face_provenance=loaded.face_provenance,
    )
    (out / "report.json").write_text(report.to_json())
    (out / "report.md").write_text(report.to_markdown())
    return report


def _load_env_and_package(env_source):
    if isinstance(env_source, VerificationEnv):
        return env_source, env_source.part_package
    env_path = Path(env_source)
    env = VerificationEnv.from_json(env_path.read_text())
    package = Path(env.part_package)
    if not package.is_absolute():
        candidate = env_path.resolve().parent / package
        if candidate.exists():
            package = candidate
    return env, str(package)
