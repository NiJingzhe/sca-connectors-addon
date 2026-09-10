"""Executable spec for the CalculiX .frd result parser.

The fixture ``tet043.frd`` is a real FEMaster v2.8.0 output (4-node tet,
fixed ROOT face, 1000 N tip load), so column widths and block layout are
ground truth, not a re-implementation guess.
"""

from pathlib import Path

import pytest

from connverify.frd import FrdBlock, FrdParseError, FrdResult, parse_frd

FIXTURE = Path(__file__).parent / "fixtures" / "tet043.frd"


@pytest.fixture(scope="module")
def frd() -> FrdResult:
    return parse_frd(FIXTURE)


class TestStructure:
    def test_node_block_is_parsed_with_coordinates(self, frd):
        assert frd.nodes[1] == pytest.approx((0.0, 0.0, 0.0))
        assert frd.nodes[2] == pytest.approx((100.0, 0.0, 0.0))
        assert frd.nodes[3] == pytest.approx((0.0, 50.0, 0.0))
        assert frd.nodes[4] == pytest.approx((0.0, 0.0, 50.0))
        assert len(frd.nodes) == 4

    def test_blocks_carry_component_names(self, frd):
        stress = frd.block("STRESS")
        assert isinstance(stress, FrdBlock)
        assert stress.components == ("SXX", "SYY", "SZZ", "SYZ", "SZX", "SXY")

    def test_element_blocks_do_not_pollute_results(self, frd):
        assert "STRESS" in frd.blocks
        assert set(frd.nodes) == {1, 2, 3, 4}

    def test_seven_component_blocks_read_the_continuation_line(self, frd):
        disp = frd.block("DISP")
        assert disp.components[0] == "D1"
        assert disp.values[2][0] == pytest.approx(8.48980e-04, rel=1e-4)
        assert disp.values[2][6] == pytest.approx(8.48980e-04, rel=1e-4)  # ALL on -2 line


class TestStress:
    def test_stress_components_by_node(self, frd):
        s = frd.stress_at(1)
        assert s.sxx_mpa == pytest.approx(2.4, rel=1e-5)
        assert s.syy_mpa == pytest.approx(1.02857, rel=1e-4)
        assert s.szz_mpa == pytest.approx(1.02857, rel=1e-4)
        assert s.syz_mpa == 0.0 and s.szx_mpa == 0.0 and s.sxy_mpa == 0.0

    def test_von_mises_uniaxial_equals_axial_minus_lateral(self, frd):
        # sigma_yy = sigma_zz here, so vM = |sxx - syy| exactly
        assert frd.von_mises_at(1) == pytest.approx(2.4 - 1.02857, rel=1e-4)

    def test_max_von_mises_reports_node_and_value(self, frd):
        node_id, value = frd.max_von_mises()
        assert node_id in frd.nodes
        assert value == pytest.approx(2.4 - 1.02857, rel=1e-4)

    def test_missing_stress_block_raises_loudly(self, frd):
        stripped = FrdResult(nodes=frd.nodes,
                              blocks={k: v for k, v in frd.blocks.items()
                                      if k != "STRESS"})
        with pytest.raises(FrdParseError) as ei:
            stripped.von_mises_at(1)
        assert "STRESS" in str(ei.value)


class TestDisplacement:
    def test_displacements_by_node(self, frd):
        d = frd.displacement_at(2)
        assert d.dx_mm == pytest.approx(8.48980e-04, rel=1e-4)
        assert d.dy_mm == pytest.approx(0.0)
        assert d.dz_mm == pytest.approx(0.0)


class TestGuards:
    def test_garbage_input_raises(self, tmp_path):
        bad = tmp_path / "bad.frd"
        bad.write_text("hello world\nnot an frd at all\n")
        with pytest.raises(FrdParseError):
            parse_frd(bad)

    def test_truncated_data_line_raises(self, tmp_path):
        text = FIXTURE.read_text().replace(
            " -1         2 8.48980E-04 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00",
            " -1         2 8.48980E-04 0.00000E+00",
        )
        bad = tmp_path / "trunc.frd"
        bad.write_text(text)
        with pytest.raises(FrdParseError):
            parse_frd(bad)
