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
