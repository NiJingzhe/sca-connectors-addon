"""Executable spec for the mesh-independence (convergence) analysis.

Pure math on recorded quantity-of-interest series — no solver needed. The
canonical numbers come from the published three-mesh audit example: meshes at
h = 20/10/5 mm (refinement ratio 2) giving 92.0 / 98.0 / 99.5, for which the
observed order is exactly 2, the Richardson limit is exactly 100.0 and the
fine-grid GCI is 0.63%.
"""

import math

import pytest

from connverify.convergence import (
    GCI_SAFETY_FACTOR,
    QoiPoint,
    analyze_convergence,
    grid_convergence_index,
    observed_order,
    relative_change_pct,
)


def series(sizes, vm, disp=None):
    """QoI points coarse -> fine (sizes descending)."""
    return tuple(
        QoiPoint(size_mm=s, node_count=100 * (i + 1), tet_count=400 * (i + 1),
                 max_von_mises_mpa=v,
                 max_displacement_mm=None if disp is None else disp[i])
        for i, (s, v) in enumerate(zip(sizes, vm))
    )


class TestRelativeChange:
    def test_symmetric_ratio_against_the_finest_value(self):
        assert relative_change_pct(99.5, 92.0) == pytest.approx(7.5 / 99.5 * 100)

    def test_zero_reference_does_not_divide_by_zero(self):
        assert relative_change_pct(1.0, 0.0) == pytest.approx(100.0)
        assert relative_change_pct(0.0, 0.0) == 0.0


class TestObservedOrder:
    def test_atlas_example_gives_exactly_two(self):
        p = observed_order(92.0, 98.0, 99.5, r=2.0)
        assert p == pytest.approx(2.0, abs=1e-9)

    def test_oscillatory_differences_return_none(self):
        assert observed_order(99.5, 92.0, 98.0, r=2.0) is None

    def test_flat_differences_return_none(self):
        assert observed_order(98.0, 98.0, 98.0, r=2.0) is None


class TestRichardsonAndGci:
    def test_atlas_example_limit_is_one_hundred(self):
        # Q_ext = 99.5 + (99.5 - 98) / (2^2 - 1)
        from connverify.convergence import richardson_extrapolated
        assert richardson_extrapolated(99.5, 98.0, p=2.0, r=2.0) == \
            pytest.approx(100.0, abs=1e-9)

    def test_atlas_example_gci_is_0p63_percent(self):
        gci = grid_convergence_index(99.5, 98.0, p=2.0, r=2.0)
        assert gci == pytest.approx(
            GCI_SAFETY_FACTOR * (1.5 / 99.5) / 3.0 * 100.0, rel=1e-3)

    def test_gci_safety_factor_is_the_standard_1p25(self):
        assert GCI_SAFETY_FACTOR == pytest.approx(1.25)


class TestAnalyzeConvergence:
    def test_converged_study_carries_full_evidence(self):
        points = series((20.0, 10.0, 5.0), (92.0, 98.0, 99.5))
        result = analyze_convergence("op", points, qoi_tolerance_pct=2.0)
        assert result.converged
        assert result.max_adjacent_delta_pct == pytest.approx(6.0 / 98.0 * 100, rel=1e-3)
        assert result.observed_order == pytest.approx(2.0, abs=1e-9)
        assert result.richardson_extrapolated_mpa == pytest.approx(100.0, abs=1e-9)
        assert result.gci_fine_pct == pytest.approx(0.628, rel=1e-2)
        assert result.extrapolation_reliable
        assert result.notes == ()

    def test_unconverged_study_fails(self):
        points = series((20.0, 10.0, 5.0), (92.0, 98.0, 110.0))
        result = analyze_convergence("op", points, qoi_tolerance_pct=2.0)
        assert not result.converged
        assert result.max_adjacent_delta_pct > 2.0

    def test_rising_peak_is_flagged_as_possible_singularity(self):
        # re-entrant-corner signature: peak stress climbs with refinement
        points = series((20.0, 10.0, 5.0), (250.0, 290.0, 335.0))
        result = analyze_convergence("op", points, qoi_tolerance_pct=2.0)
        assert not result.converged
        assert any("singular" in n.lower() for n in result.notes)
        assert not result.extrapolation_reliable

    def test_identical_values_converge_without_extrapolation(self):
        points = series((20.0, 10.0, 5.0), (98.0, 98.0, 98.0))
        result = analyze_convergence("op", points, qoi_tolerance_pct=1.0)
        assert result.converged
        assert result.max_adjacent_delta_pct == 0.0
        assert result.observed_order is None
        assert not result.extrapolation_reliable

    def test_oscillatory_series_is_flagged(self):
        points = series((20.0, 10.0, 5.0), (99.5, 92.0, 98.0))
        result = analyze_convergence("op", points, qoi_tolerance_pct=50.0)
        assert result.converged  # deltas within a loose tolerance...
        assert not result.extrapolation_reliable  # ...but evidence is noted
        assert any("oscillat" in n.lower() for n in result.notes)

    def test_two_meshes_converge_but_cannot_extrapolate(self):
        points = series((10.0, 5.0), (98.0, 99.0))
        result = analyze_convergence("op", points, qoi_tolerance_pct=2.0)
        assert result.converged
        assert result.observed_order is None
        assert result.richardson_extrapolated_mpa is None
        assert result.gci_fine_pct is None
        assert any("third" in n for n in result.notes)

    def test_non_geometric_family_skips_extrapolation(self):
        points = series((24.0, 16.0, 8.0), (92.0, 98.0, 99.5))
        result = analyze_convergence("op", points, qoi_tolerance_pct=10.0)
        assert result.observed_order is None
        assert any("geometric" in n.lower() for n in result.notes)

    def test_missing_data_at_one_size_is_not_convergence(self):
        points = series((20.0, 10.0, 5.0), (92.0, None, 99.5))
        result = analyze_convergence("op", points, qoi_tolerance_pct=2.0)
        assert not result.converged
        assert any("no data" in n.lower() for n in result.notes)

    def test_displacement_also_gates_convergence(self):
        points = series((20.0, 10.0, 5.0), (98.0, 98.5, 98.6),
                        disp=(1.90, 1.98, 2.30))
        result = analyze_convergence("op", points, qoi_tolerance_pct=2.0)
        assert not result.converged
        assert any("displacement" in n.lower() for n in result.notes)

    def test_points_are_sorted_coarse_to_fine_regardless_of_input(self):
        points = tuple(reversed(series((20.0, 10.0, 5.0), (92.0, 98.0, 99.5))))
        result = analyze_convergence("op", points, qoi_tolerance_pct=2.0)
        assert [p.size_mm for p in result.points] == [20.0, 10.0, 5.0]
        assert result.converged

    def test_to_dict_is_json_ready(self):
        points = series((20.0, 10.0, 5.0), (92.0, 98.0, 99.5))
        payload = analyze_convergence("op", points, qoi_tolerance_pct=2.0).to_dict()
        import json
        json.dumps(payload)
        assert payload["name"] == "op"
        assert payload["converged"] is True
        assert len(payload["points"]) == 3


class TestQoiExtraction:
    def test_qoi_point_from_synthetic_frd(self):
        from connverify.frd import FrdBlock, FrdResult
        from connverify.convergence import qoi_point_from_frd

        nodes = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)}
        frd = FrdResult(nodes=nodes, blocks={
            "STRESS": FrdBlock("STRESS",
                               ("SXX", "SYY", "SZZ", "SYZ", "SZX", "SXY"),
                               {1: (100.0, 0, 0, 0, 0, 0),
                                2: (60.0, 0, 0, 0, 0, 0)}),
            "DISP": FrdBlock("DISP", ("D1", "D2", "D3"),
                             {1: (0.003, 0.0, 0.004), 2: (0.0, 0.0, 0.0)}),
        })
        point = qoi_point_from_frd(size_mm=8.0, frd=frd,
                                   yield_strength_mpa=355.0)
        assert point.max_von_mises_mpa == pytest.approx(100.0)
        assert point.safety_factor == pytest.approx(3.55)
        assert point.max_displacement_mm == pytest.approx(0.005)
        assert point.size_mm == pytest.approx(8.0)

    def test_qoi_point_degrades_without_disp_block(self):
        from connverify.frd import FrdBlock, FrdResult
        from connverify.convergence import qoi_point_from_frd

        frd = FrdResult(nodes={1: (0.0, 0.0, 0.0)}, blocks={
            "STRESS": FrdBlock("STRESS",
                               ("SXX", "SYY", "SZZ", "SYZ", "SZX", "SXY"),
                               {1: (100.0, 0, 0, 0, 0, 0)}),
        })
        point = qoi_point_from_frd(size_mm=8.0, frd=frd, yield_strength_mpa=355.0)
        assert point.max_displacement_mm is None
        assert point.max_von_mises_mpa == pytest.approx(100.0)
