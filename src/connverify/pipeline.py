"""One-call orchestration: environment + package -> checks -> mesh -> solve
-> geometry-directed report.

Run directory layout (all durable artifacts):

::

    <out_dir>/
      decks/<CASE>.inp        generated FEMaster decks (single-mesh run)
      decks/h<SIZE>/<CASE>.inp  per-size decks when env.mesh_study is set
      decks/.../*.frd/.res    solver outputs (next to their decks)
      stress_<CASE>.png       von Mises contour per solved case (finest mesh)
      convergence.png         QoI-vs-size study plot (mesh_study runs only)
      report.json             machine contract
      report.md               agent-readable rendering

Solve discipline: a mesh that fails the deterministic quality gate
(inverted elements, aspect-ratio hard limit) is never solved — every case
fails loudly instead of producing numerically meaningless stresses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .conncheck import check_interface
from .convergence import CaseConvergence, analyze_convergence, qoi_point_from_frd
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
    if env.mesh_study is not None and mesh_size_mm is not None:
        raise ValueError(
            "pass mesh_size_mm= or env.mesh_study, not both — the study owns "
            "the mesh family it verifies"
        )

    loaded = load_part(package_path)
    require_interfaces(loaded, env.required_interface_names())

    conn_results = tuple(check_interface(loaded, i) for i in env.interfaces)
    envelope_results = check_envelopes(loaded, env.envelopes)

    study = env.mesh_study
    sizes: Tuple[Optional[float], ...] = \
        study.sizes_mm if study is not None else (mesh_size_mm,)
    meshes: List[Tuple[Optional[float], object]] = [
        (size, mesh_part(loaded, mesh_size_mm=size)) for size in sizes]

    out = Path(out_dir)
    deck_dir = out / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)

    render_facts = None
    if render_tag_check:
        from .render import render_tag_review

        render_facts = render_tag_review(
            meshes[-1][1], out / "tag_review.png",
            title=f"tag review — {env.name}",
        )

    gate = next(((size, mesh) for size, mesh in meshes
                 if mesh.quality is not None and not mesh.quality.passed), None)

    outcomes_by_size: List[Dict[str, CaseOutcome]] = []
    qoi_by_size: List[Dict[str, object]] = []
    if gate is not None:
        _size, gate_mesh = gate
        for _size_i, _mesh_i in meshes:
            per_case = {
                case.name: CaseOutcome(
                    name=case.name, frd=None,
                    error="mesh quality gate failed — no solve attempted: "
                    + "; ".join(gate_mesh.quality.failures),
                    duration_s=None)
                for case in env.load_cases
            }
            outcomes_by_size.append(per_case)
            qoi_by_size.append({})
    else:
        for size, mesh in meshes:
            decks = generate_decks(mesh, env)
            case_dir = deck_dir if study is None else deck_dir / f"h{size:g}"
            case_dir.mkdir(parents=True, exist_ok=True)
            outcomes: Dict[str, CaseOutcome] = {}
            qois: Dict[str, object] = {}
            for case in env.load_cases:
                deck_path = case_dir / f"{case.name}.inp"
                deck_path.write_text(decks[case.name])
                try:
                    solved = run_solve(str(deck_path), binary=femaster,
                                       timeout_s=timeout_s, ncpus=ncpus)
                    frd = parse_frd(solved.frd_path) if solved.frd_path else None
                    outcomes[case.name] = CaseOutcome(
                        name=case.name, frd=frd,
                        error=None if frd is not None else
                        "solver produced no .frd result file",
                        duration_s=solved.duration_s,
                    )
                    if frd is not None:
                        qois[case.name] = qoi_point_from_frd(
                            size_mm=mesh.target_size_mm or 0.0, frd=frd,
                            yield_strength_mpa=env.material.yield_strength_mpa,
                            node_count=mesh.node_count,
                            tet_count=mesh.tet_count)
                except Exception as exc:  # solver/deck errors become case failures
                    outcomes[case.name] = CaseOutcome(
                        name=case.name, frd=None, error=str(exc), duration_s=None)
            outcomes_by_size.append(outcomes)
            qoi_by_size.append(qois)

    finest_mesh = meshes[-1][1]
    finest_outcomes = outcomes_by_size[-1]

    for outcome in finest_outcomes.values():
        if outcome.frd is None:
            continue
        from .render import render_stress_contour

        render_stress_contour(
            finest_mesh, outcome.frd, out / f"stress_{outcome.name}.png",
            title=f"von Mises — {env.name} / {outcome.name}")

    convergence: Tuple[CaseConvergence, ...] = ()
    if study is not None:
        convergence = tuple(
            analyze_convergence(
                case.name,
                [qoi_by_size[i][case.name] for i in range(len(meshes))
                 if case.name in qoi_by_size[i]],
                qoi_tolerance_pct=study.qoi_tolerance_pct)
            for case in env.load_cases)
        from .render import render_convergence

        render_convergence(
            convergence, out / "convergence.png",
            title=f"mesh independence — {env.name}",
            tolerance_pct=study.qoi_tolerance_pct)

    report = build_report(
        env=env, mesh=finest_mesh,
        conn_results=conn_results, envelope_results=envelope_results,
        outcomes=tuple(finest_outcomes[c.name] for c in env.load_cases),
        face_provenance=loaded.face_provenance,
        meshes=tuple(meshes), convergence=convergence, mesh_study=study,
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
