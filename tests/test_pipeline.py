"""End-to-end pipeline test: real package -> checks -> mesh -> deck -> REAL
FEMaster solve -> .frd -> report. Uses the vendored FEMaster binary (resolved
automatically); skips loudly if no binary is available.
"""

import json
import os

import pytest

from connverify.env import (
    ConnectionMethod, ForceLoad, Interface, KeepOutBox, LoadCase, Material,
    VerificationEnv,
)
from connverify.pipeline import verify
from connverify.solver import locate_femaster

requires_femaster = pytest.mark.skipif(
    os.environ.get("CONNVERIFY_SKIP_FEMASTER") == "1",
    reason="CONNVERIFY_SKIP_FEMASTER=1",
)


@pytest.fixture(scope="module")
def env_path(probe_box_pkg, tmp_path_factory):
    """A steel plate bolted via mount_face and pressed on load_pad."""
    env = VerificationEnv(
        name="plate mount e2e",
        part_package=os.path.basename(probe_box_pkg),
        material=Material(
            name="steel_s355", youngs_modulus_mpa=210000.0, poisson_ratio=0.3,
            yield_strength_mpa=355.0, density_t_per_mm3=7.85e-9),
        interfaces=[
            Interface(name="mount_face", method=ConnectionMethod.BOLTED),
            Interface(name="load_pad", method=ConnectionMethod.CONTACT),
        ],
        load_cases=[
            LoadCase(name="press", loads=[ForceLoad(target="load_pad", fz_n=-1000.0)]),
        ],
        envelopes=[
            KeepOutBox(name="clear_above", min_corner_mm=(100.0, 100.0, 100.0),
                       max_corner_mm=(200.0, 200.0, 200.0)),
        ],
        safety_factor_required=1.5,
    )
    pkg_dir = os.path.dirname(probe_box_pkg)
    path = os.path.join(pkg_dir, "env_e2e.json")
    with open(path, "w") as handle:
        handle.write(env.to_json())
    return path


@requires_femaster
def test_end_to_end_passing_verification(env_path, tmp_path):
    out_dir = tmp_path / "verify_out"
    report = verify(env_path, out_dir=out_dir, mesh_size_mm=12.0)

    assert report.verdict == "pass"
    case = report.cases[0]
    assert case.error is None
    assert case.max_von_mises_mpa > 0.0
    assert case.safety_factor > 1.5
    assert case.hotspots and case.hotspots[0].location_mm is not None

    # durable artifacts: decks, results, report json + markdown
    assert (out_dir / "report.json").is_file()
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "tag_review.png").is_file()   # visual tag check artifact
    assert (out_dir / "stress_PRESS.png").is_file()  # FEM contour artifact
    deck = out_dir / "decks" / "PRESS.inp"
    assert deck.is_file()
    payload = json.loads((out_dir / "report.json").read_text())
    assert payload["verdict"] == "pass"
    assert payload["load_cases"][0]["name"] == "press"
    markdown = (out_dir / "report.md").read_text()
    assert "press" in markdown and "von Mises" in markdown


def test_missing_interface_stops_before_meshing(probe_box_pkg, tmp_path):
    env = VerificationEnv(
        name="bad", part_package=probe_box_pkg,
        material=Material(name="s", youngs_modulus_mpa=210000.0, poisson_ratio=0.3,
                          yield_strength_mpa=355.0, density_t_per_mm3=7.85e-9),
        interfaces=[Interface(name="ghost_face", method=ConnectionMethod.FIXED)],
        load_cases=[LoadCase(name="l", loads=[ForceLoad(target="ghost_face",
                                                        fz_n=-1.0)])],
    )
    from connverify.package_reader import MissingInterfaceError
    with pytest.raises(MissingInterfaceError) as ei:
        verify(env, out_dir=tmp_path / "nope")
    assert "interface.ghost_face" in str(ei.value)


@requires_femaster
def test_end_to_end_mesh_study_writes_convergence_evidence(env_path, tmp_path):
    """Three-mesh study on the probe box. Physically, peak von Mises at the
    clamped mount face rises with refinement (stress concentration), so a
    20% tolerance converges and passes, while a 5% tolerance must FAIL with
    the singularity surfaced — both outcomes carry full evidence."""
    import json
    from pathlib import Path

    from connverify.env import VerificationEnv

    payload = json.loads(Path(env_path).read_text())
    payload["name"] = "plate mount study e2e"
    payload["mesh_study"] = {"sizes_mm": [9.0, 6.0, 4.0],
                             "qoi_tolerance_pct": 20.0}
    env = VerificationEnv.from_json(json.dumps(payload))
    pkg_dir = os.path.dirname(env_path)
    study_path = os.path.join(pkg_dir, "env_study_e2e.json")
    with open(study_path, "w") as handle:
        handle.write(env.to_json())

    out_dir = tmp_path / "verify_study"
    report = verify(study_path, out_dir=out_dir)

    aq = report.analysis_quality
    assert aq["convergence"]["performed"] is True
    case = aq["convergence"]["cases"][0]
    assert len(case["points"]) == 3
    assert case["points"][0]["size_mm"] == pytest.approx(9.0)   # coarse first
    assert case["points"][2]["size_mm"] == pytest.approx(4.0)   # fine last
    sizes = [p["node_count"] for p in case["points"]]
    assert sizes[0] < sizes[1] < sizes[2]                        # real refinement
    assert case["converged"] is True

    assert (out_dir / "convergence.png").is_file()
    for size in (9.0, 6.0, 4.0):
        assert (out_dir / "decks" / f"h{size:g}" / "PRESS.inp").is_file()
    assert (out_dir / "stress_PRESS.png").is_file()

    written = json.loads((out_dir / "report.json").read_text())
    assert written["analysis_quality"]["convergence"]["passed"] is True
    markdown = (out_dir / "report.md").read_text()
    assert "## Mesh independence" in markdown
    assert "ΔQ" in markdown
    assert report.verdict == "pass"


@requires_femaster
def test_end_to_end_unconverged_study_fails_with_singularity_note(env_path,
                                                                  tmp_path):
    import json
    from pathlib import Path

    from connverify.env import VerificationEnv

    payload = json.loads(Path(env_path).read_text())
    payload["mesh_study"] = {"sizes_mm": [9.0, 6.0, 4.0],
                             "qoi_tolerance_pct": 5.0}
    env = VerificationEnv.from_json(json.dumps(payload))
    pkg_dir = os.path.dirname(env_path)
    tight_path = os.path.join(pkg_dir, "env_study_tight.json")
    with open(tight_path, "w") as handle:
        handle.write(env.to_json())

    report = verify(tight_path, out_dir=tmp_path / "verify_tight")
    assert report.verdict == "fail"
    assert any("did not converge" in item for item in report.feedback)
    assert any("singular" in item.lower() for item in report.feedback)
    case = report.analysis_quality["convergence"]["cases"][0]
    assert case["converged"] is False
    assert case["max_adjacent_delta_pct"] > 5.0


@requires_femaster
def test_study_and_mesh_size_kwarg_conflict_is_rejected(env_path, tmp_path):
    import json
    from pathlib import Path

    payload = json.loads(Path(env_path).read_text())
    payload["mesh_study"] = {"sizes_mm": [9.0, 6.0, 4.0], "qoi_tolerance_pct": 5.0}
    with open(env_path, "w") as handle:
        handle.write(json.dumps(payload))
    try:
        with pytest.raises(ValueError, match="not both"):
            verify(env_path, out_dir=tmp_path / "x", mesh_size_mm=10.0)
    finally:
        payload.pop("mesh_study")
        with open(env_path, "w") as handle:
            handle.write(json.dumps(payload))
