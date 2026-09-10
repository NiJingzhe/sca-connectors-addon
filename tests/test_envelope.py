"""Executable spec for keep-out envelope checks (OCC boolean intersection)."""

import pytest

from connverify.envelope import check_envelope, check_envelopes
from connverify.env import KeepOutBox
from connverify.package_reader import load_part


@pytest.fixture(scope="module")
def loaded(probe_box_pkg):
    return load_part(probe_box_pkg)


class TestKeepOutBoxes:
    def test_disjoint_box_passes_with_zero_intersection(self, loaded):
        keepout = KeepOutBox(name="clear_zone",
                             min_corner_mm=(100.0, 100.0, 100.0),
                             max_corner_mm=(200.0, 200.0, 200.0))
        result = check_envelope(loaded, keepout)
        assert result.passed
        assert result.intersection_volume_mm3 == pytest.approx(0.0)

    def test_overlapping_box_fails_with_intersection_volume(self, loaded):
        # part box x[0,60] y[0,40] z[0,30]; overlap = [0,5]x[0,5]x[25,30] = 125 mm³
        keepout = KeepOutBox(name="motor_housing",
                             min_corner_mm=(-5.0, -5.0, 25.0),
                             max_corner_mm=(5.0, 5.0, 35.0))
        result = check_envelope(loaded, keepout)
        assert not result.passed
        assert result.intersection_volume_mm3 == pytest.approx(125.0, rel=0.01)

    def test_result_carries_penetration_bbox(self, loaded):
        keepout = KeepOutBox(name="motor_housing",
                             min_corner_mm=(-5.0, -5.0, 25.0),
                             max_corner_mm=(5.0, 5.0, 35.0))
        result = check_envelope(loaded, keepout)
        assert result.intersection_bbox_mm is not None
        (xmin, ymin, zmin), (xmax, ymax, zmax) = result.intersection_bbox_mm
        assert xmax == pytest.approx(5.0, abs=1e-6)
        assert zmin == pytest.approx(25.0, abs=1e-6)

    def test_envelopes_checked_together(self, loaded):
        results = check_envelopes(loaded, [
            KeepOutBox(name="clear", min_corner_mm=(100, 100, 100),
                       max_corner_mm=(200, 200, 200)),
            KeepOutBox(name="hit", min_corner_mm=(-5, -5, 25),
                       max_corner_mm=(5, 5, 35)),
        ])
        assert [r.passed for r in results] == [True, False]
