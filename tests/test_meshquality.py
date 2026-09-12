"""Executable spec for the deterministic mesh quality gate.

Metrics are pure geometry (numpy, vectorized) so they are exactly
reproducible: signed volume detects inverted elements, the normalized radius
ratio R/(3r) is the aspect ratio (1.0 = regular tetrahedron), and gmsh's
minSICN (signed inverse condition number) cross-checks the mesher's own view.
"""

import math

import pytest

from connverify.mesh_model import Tet
from connverify.meshquality import (
    ASPECT_RATIO_FAIL,
    compute_mesh_quality,
    tet_aspect_ratios,
    tet_signed_volumes,
)

# Regular tetrahedron with unit edge (exact coordinates).
_A = (0.0, 0.0, 0.0)
_B = (1.0, 0.0, 0.0)
_C = (0.5, math.sqrt(3.0) / 2.0, 0.0)
_D = (0.5, math.sqrt(3.0) / 6.0, math.sqrt(2.0 / 3.0))
_REGULAR_VOLUME = math.sqrt(2.0) / 12.0  # a^3 / (6 sqrt(2)) with a = 1


def regular_nodes():
    return {1: _A, 2: _B, 3: _C, 4: _D}


class TestTetGeometry:
    def test_regular_tet_has_unit_aspect_ratio(self):
        ar = tet_aspect_ratios(regular_nodes(), (Tet(1, (1, 2, 3, 4)),))
        assert ar[0] == pytest.approx(1.0, rel=1e-9)

    def test_regular_tet_volume_is_exact(self):
        v = tet_signed_volumes(regular_nodes(), (Tet(1, (1, 2, 3, 4)),))
        assert v[0] == pytest.approx(_REGULAR_VOLUME, rel=1e-9)

    def test_swapped_nodes_make_the_volume_negative(self):
        v = tet_signed_volumes(regular_nodes(), (Tet(1, (2, 1, 3, 4)),))
        assert v[0] == pytest.approx(-_REGULAR_VOLUME, rel=1e-9)

    def test_stretched_tet_aspect_ratio_grows(self):
        nodes = {1: (0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0),
                 3: (0.0, 1.0, 0.0), 4: (0.0, 0.0, 1.0)}
        ar = tet_aspect_ratios(nodes, (Tet(1, (1, 2, 3, 4)),))
        assert ar[0] > 5.0

    def test_metrics_are_vectorized_and_order_stable(self):
        nodes = {**regular_nodes(), 5: (3.0, 0.0, 0.0), 6: (3.0, 1.0, 0.0),
                 7: (3.0, 0.0, 1.0), 8: (4.0, 0.0, 0.5)}
        tets = (Tet(1, (1, 2, 3, 4)), Tet(2, (5, 6, 7, 8)))
        again = (Tet(1, (1, 2, 3, 4)), Tet(2, (5, 6, 7, 8)))
        assert list(tet_signed_volumes(nodes, tets)) == \
            list(tet_signed_volumes(nodes, again))
        assert len(tet_aspect_ratios(nodes, tets)) == 2


class TestQualityGate:
    def test_good_mesh_passes_with_full_stats(self):
        stats = compute_mesh_quality(regular_nodes(), (Tet(1, (1, 2, 3, 4)),))
        assert stats.passed
        assert stats.failures == ()
        assert stats.tet_count == 1
        assert stats.inverted_tets == 0
        assert stats.aspect_ratio_max == pytest.approx(1.0, rel=1e-9)
        assert stats.volume_min_mm3 == pytest.approx(_REGULAR_VOLUME, rel=1e-9)

    def test_inverted_element_fails_the_gate(self):
        stats = compute_mesh_quality(regular_nodes(), (Tet(1, (2, 1, 3, 4)),))
        assert not stats.passed
        assert stats.inverted_tets == 1
        assert any("inverted" in f for f in stats.failures)

    def test_sliver_fails_with_location(self):
        # four nodes nearly coplanar: huge aspect ratio
        nodes = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0),
                 3: (0.0, 1.0, 0.0), 4: (1.0, 1.0, 1e-6)}
        stats = compute_mesh_quality(nodes, (Tet(1, (1, 2, 3, 4)),))
        assert not stats.passed
        assert stats.aspect_ratio_max > ASPECT_RATIO_FAIL
        assert any("aspect ratio" in f for f in stats.failures)
        assert all(math.isfinite(v) for v in stats.worst_ar_element_centroid_mm)

    def test_warning_band_is_reported_but_passes(self):
        # one stretched tet lands in the 5 < AR < 10 band: warning, not fail
        nodes = {1: (0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0),
                 3: (0.0, 1.0, 0.0), 4: (0.0, 0.0, 1.0),
                 5: _A, 6: _B, 7: _C, 8: _D}
        tets = (Tet(1, (1, 2, 3, 4)), Tet(2, (5, 6, 7, 8)))
        stats = compute_mesh_quality(nodes, tets)
        assert 5.0 < stats.aspect_ratio_max < ASPECT_RATIO_FAIL
        assert stats.passed
        assert stats.warnings

    def test_sicn_values_are_carried_into_stats(self):
        stats = compute_mesh_quality(
            regular_nodes(), (Tet(1, (1, 2, 3, 4)),), sicn=(0.9,))
        assert stats.sicn_min == pytest.approx(0.9)
        assert stats.sicn_mean == pytest.approx(0.9)

    def test_low_sicn_warns(self):
        stats = compute_mesh_quality(
            regular_nodes(), (Tet(1, (1, 2, 3, 4)),), sicn=(0.01,))
        assert stats.passed  # warn-level, not a hard failure
        assert any("SICN" in w for w in stats.warnings)

    def test_empty_mesh_fails_loudly(self):
        stats = compute_mesh_quality(regular_nodes(), ())
        assert not stats.passed
        assert any("no elements" in f for f in stats.failures)

    def test_stats_are_deterministic(self):
        tets = (Tet(1, (1, 2, 3, 4)),)
        a = compute_mesh_quality(regular_nodes(), tets)
        b = compute_mesh_quality(regular_nodes(), tets)
        assert a == b

    def test_to_dict_is_json_ready(self):
        stats = compute_mesh_quality(regular_nodes(), (Tet(1, (1, 2, 3, 4)),))
        payload = stats.to_dict()
        import json
        json.dumps(payload)  # must serialize
        assert payload["tet_count"] == 1
        assert payload["passed"] is True
        assert "aspect_ratio_max" in payload


class TestRealMesh:
    def test_pipeline_mesh_carries_quality(self, probe_box_pkg):
        from connverify.meshing import mesh_part
        from connverify.package_reader import load_part

        mesh = mesh_part(load_part(probe_box_pkg), mesh_size_mm=10.0)
        assert mesh.quality is not None
        assert mesh.quality.passed
        assert mesh.quality.inverted_tets == 0
        assert mesh.quality.aspect_ratio_max < ASPECT_RATIO_FAIL
        q = mesh.quality
        assert q.aspect_ratio_min <= q.aspect_ratio_mean <= q.aspect_ratio_max
        assert q.aspect_ratio_p95 <= q.aspect_ratio_max

    def test_gmsh_sicn_is_sampled(self, probe_box_pkg):
        from connverify.meshing import mesh_part
        from connverify.package_reader import load_part

        mesh = mesh_part(load_part(probe_box_pkg), mesh_size_mm=10.0)
        assert mesh.quality.sicn_min is not None
        assert mesh.quality.sicn_min > 0.0
        assert mesh.quality.sicn_mean >= mesh.quality.sicn_min

    def test_target_size_is_recorded(self, probe_box_pkg):
        from connverify.meshing import mesh_part
        from connverify.package_reader import load_part

        mesh = mesh_part(load_part(probe_box_pkg), mesh_size_mm=10.0)
        assert mesh.target_size_mm == pytest.approx(10.0)
        auto = mesh_part(load_part(probe_box_pkg))
        assert auto.target_size_mm is not None and auto.target_size_mm > 0.0
