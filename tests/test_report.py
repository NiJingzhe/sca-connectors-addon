"""Executable spec for the geometry-directed feedback report.

The report is the addon's contract with agents: every failing fact must name
WHERE in the geometry it lives (interface, BREP feature provenance, or
coordinates) and WHAT to change — never a bare pass/fail.
"""

import pytest
from pathlib import Path

from connverify.conncheck import InterfaceCheckResult
from connverify.envelope import EnvelopeCheckResult
from connverify.env import (
    ConnectionMethod, Interface, LoadCase, Material, ForceLoad, VerificationEnv,
)
from connverify.frd import parse_frd
from connverify.report import CaseOutcome, build_report

from tests.test_mesh_model import single_tet_mesh

FIXTURE_FRD = Path(__file__).parent / "fixtures" / "tet043.frd"

PROVENANCE = {
    "face_mount": {"graph_id": "probe", "node_id": "node_00000001", "output_slot": 0},
    "face_load": {"graph_id": "probe", "node_id": "node_00000001", "output_slot": 0},
}


def make_env(**over):
    kw = dict(
        name="tet probe",
        part_package="out/tet.scadpkg",
        material=Material(
            name="steel_s355", youngs_modulus_mpa=210000.0, poisson_ratio=0.3,
            yield_strength_mpa=355.0, density_t_per_mm3=7.85e-9),
        interfaces=[
            Interface(name="mount", method=ConnectionMethod.BOLTED),
            Interface(name="load", method=ConnectionMethod.CONTACT),
        ],
        load_cases=[LoadCase(name="op", loads=[ForceLoad(target="load", fz_n=-300.0)])],
        envelopes=[],
        safety_factor_required=1.5,
    )
    kw.update(over)
    return VerificationEnv(**kw)


def good_conn_results():
    return (
        InterfaceCheckResult(
            interface_name="mount", method=ConnectionMethod.BOLTED, passed=True,
            face_count=1, total_area_mm2=0.5, planarity_deviation_mm=0.0,
            normal=(0.0, 0.0, -1.0), centroid_mm=(0.0, 0.0, 0.0),
            min_area_required_mm2=None, messages=(),
        ),
        InterfaceCheckResult(
            interface_name="load", method=ConnectionMethod.CONTACT, passed=True,
            face_count=1, total_area_mm2=0.5, planarity_deviation_mm=0.0,
            normal=None, centroid_mm=None, min_area_required_mm2=None, messages=(),
        ),
    )


def ok_outcome():
    return CaseOutcome(name="op", frd=parse_frd(FIXTURE_FRD), error=None,
                       duration_s=0.013)


class TestPassingReport:
    @pytest.fixture()
    def report(self):
        return build_report(
            env=make_env(), mesh=single_tet_mesh(),
            conn_results=good_conn_results(), envelope_results=(),
            outcomes=[ok_outcome()], face_provenance=PROVENANCE,
        )

    def test_verdict_passes_when_everything_is_green(self, report):
        assert report.verdict == "pass"

    def test_case_carries_numbers(self, report):
        case = report.cases[0]
        # fixture: vM = 2.4 - 1.02857 on every node; SF = 355 / vM
        assert case.max_von_mises_mpa == pytest.approx(2.4 - 1.02857, rel=1e-4)
        assert case.safety_factor == pytest.approx(355.0 / (2.4 - 1.02857), rel=1e-4)
        assert case.max_displacement_mm == pytest.approx(8.4898e-04, rel=1e-3)
        assert case.passed

    def test_hotspot_is_attributed_to_interface_and_feature(self, report):
        hotspot = report.cases[0].hotspots[0]
        assert hotspot.von_mises_mpa == pytest.approx(2.4 - 1.02857, rel=1e-4)
        assert hotspot.location_mm is not None
        assert hotspot.on_interface == "mount"  # node 1 is on the mount set
        assert hotspot.owning_feature["node_id"] == "node_00000001"

    def test_json_is_deterministic_and_schema_stamped(self, report):
        assert report.to_json() == build_report(
            env=make_env(), mesh=single_tet_mesh(),
            conn_results=good_conn_results(), envelope_results=(),
            outcomes=[ok_outcome()], face_provenance=PROVENANCE,
        ).to_json()
        import json
        payload = json.loads(report.to_json())
        assert payload["schema_version"] == "1.0"
        assert payload["verdict"] == "pass"

    def test_markdown_mentions_verdict_and_cases(self, report):
        md = report.to_markdown()
        assert "PASS" in md or "pass" in md
        assert "op" in md
        assert "von Mises" in md or "von_mises" in md


class TestFailingPaths:
    def test_low_safety_factor_fails_with_actionable_suggestion(self):
        env = make_env(material=Material(
            name="weak", youngs_modulus_mpa=210000.0, poisson_ratio=0.3,
            yield_strength_mpa=2.0, density_t_per_mm3=7.85e-9))
        report = build_report(
            env=env, mesh=single_tet_mesh(),
            conn_results=good_conn_results(), envelope_results=(),
            outcomes=[ok_outcome()], face_provenance=PROVENANCE,
        )
        assert report.verdict == "fail"
        case = report.cases[0]
        assert not case.passed
        assert case.safety_factor < 1.5
        joined = " ".join(case.suggestions)
        assert "mount" in joined  # hotspot lived on the mount interface

    def test_envelope_violation_appears_in_feedback_with_numbers(self):
        env_result = EnvelopeCheckResult(
            name="motor_housing", passed=False, intersection_volume_mm3=125.0,
            intersection_bbox_mm=((0.0, 0.0, 25.0), (5.0, 5.0, 30.0)),
        )
        report = build_report(
            env=make_env(), mesh=single_tet_mesh(),
            conn_results=good_conn_results(), envelope_results=(env_result,),
            outcomes=[ok_outcome()], face_provenance=PROVENANCE,
        )
        assert report.verdict == "fail"
        text = " ".join(report.feedback)
        assert "motor_housing" in text
        assert "125" in text

    def test_connection_failure_surfaces_its_messages(self):
        conn = list(good_conn_results())
        conn[0] = InterfaceCheckResult(
            interface_name="mount", method=ConnectionMethod.BOLTED, passed=False,
            face_count=1, total_area_mm2=100.0, planarity_deviation_mm=0.4,
            normal=(0, 0, -1), centroid_mm=(0, 0, 0),
            min_area_required_mm2=None,
            messages=("mating face not planar: deviation 0.4000 mm exceeds tolerance 0.1000 mm",),
        )
        report = build_report(
            env=make_env(), mesh=single_tet_mesh(),
            conn_results=tuple(conn), envelope_results=(),
            outcomes=[ok_outcome()], face_provenance=PROVENANCE,
        )
        assert report.verdict == "fail"
        assert any("planar" in item for item in report.feedback)

    def test_solver_error_marks_case_failed_and_names_the_error(self):
        outcome = CaseOutcome(name="op", frd=None,
                              error="FEMaster exited with code 3", duration_s=None)
        report = build_report(
            env=make_env(), mesh=single_tet_mesh(),
            conn_results=good_conn_results(), envelope_results=(),
            outcomes=[outcome], face_provenance=PROVENANCE,
        )
        assert report.verdict == "fail"
        assert report.cases[0].error == "FEMaster exited with code 3"
        assert not report.cases[0].passed


class TestHotspotRanking:
    def test_multiple_hotspots_sorted_by_stress_desc(self, report=None):
        report = build_report(
            env=make_env(), mesh=single_tet_mesh(),
            conn_results=good_conn_results(), envelope_results=(),
            outcomes=[ok_outcome()], face_provenance=PROVENANCE,
        )
        stresses = [h.von_mises_mpa for h in report.cases[0].hotspots]
        assert stresses == sorted(stresses, reverse=True)
