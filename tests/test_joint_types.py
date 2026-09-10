"""Executable spec for the mechanical connection-type registry.

Taxonomy anchor (research 2026-09-10, Wikipedia/Fastener + RoyMech + AWS):
- detachable (可拆卸): threaded (bolt/stud/screw), pins, keys, splines
- semi-permanent: interference / transition fits (ISO 286)
- permanent (不可拆): welding, riveting, adhesives
Design-rule anchors: bolt layout BS 5950 / EN 1993-1-8 (pitch ≥ 2.5d,
edge ≥ 1.5d, floor 1.2·d0, full strength 2d); fillet weld minimums
AWS D1.1 Table 7.7 + formula leg ≥ 1.5·√t; ISO 4014 hex-head geometry;
EN 1090-2 / ISO 273 clearance holes.
"""

import pytest

from connverify.env import (
    ConnectionMethod, EnvValidationError, Interface, LoadCase, Material,
    ForceLoad, VerificationEnv,
)
from connverify.joint_types import (
    AdhesiveSpec,
    BearingSeatSpec,
    BoltedTappedSpec,
    BoltedThroughSpec,
    ClampedSpec,
    InterferenceSpec,
    JointKind,
    KeyedSpec,
    PinnedSpec,
    RivetedSpec,
    WeldedFilletSpec,
    hex_head_dimensions,
    min_fillet_leg_mm,
    it7_tolerance_mm,
)


def make_material(**over):
    kw = dict(name="steel_s355", youngs_modulus_mpa=210000.0, poisson_ratio=0.3,
              yield_strength_mpa=355.0, density_t_per_mm3=7.85e-9)
    kw.update(over)
    return Material(**kw)


class TestTaxonomy:
    def test_every_kind_declares_its_permanence_class(self):
        from connverify.joint_types import PERMANENT_KINDS, SEMI_PERMANENT_KINDS
        assert JointKind.WELDED_FILLET in PERMANENT_KINDS
        assert JointKind.RIVETED in PERMANENT_KINDS
        assert JointKind.ADHESIVE in PERMANENT_KINDS
        assert JointKind.INTERFERENCE in SEMI_PERMANENT_KINDS
        assert JointKind.BOLTED_THROUGH not in PERMANENT_KINDS
        assert JointKind.PINNED not in PERMANENT_KINDS | SEMI_PERMANENT_KINDS

    def test_which_kinds_constrain_the_part_for_fem(self):
        from connverify.joint_types import kind_constrains
        assert kind_constrains(JointKind.BOLTED_THROUGH)
        assert kind_constrains(JointKind.WELDED_FILLET)
        assert kind_constrains(JointKind.CLAMPED)
        assert kind_constrains(JointKind.INTERFERENCE)
        assert kind_constrains(JointKind.PINNED)
        assert not kind_constrains(JointKind.CONTACT_PAD)
        assert not kind_constrains(JointKind.BEARING_SEAT)
        assert not kind_constrains(JointKind.KEYED)

    def test_every_kind_maps_to_a_fem_boundary_condition(self):
        from connverify.joint_types import KIND_METHOD
        assert set(KIND_METHOD) == set(JointKind)
        assert KIND_METHOD[JointKind.BOLTED_THROUGH] is ConnectionMethod.BOLTED
        assert KIND_METHOD[JointKind.WELDED_FILLET] is ConnectionMethod.WELDED
        assert KIND_METHOD[JointKind.CLAMPED] is ConnectionMethod.FIXED
        assert KIND_METHOD[JointKind.CONTACT_PAD] is ConnectionMethod.CONTACT


class TestSpecs:
    def test_bolted_through_carries_layout_rule_limits(self):
        spec = BoltedThroughSpec(nominal_diameter_mm=10.0, bolt_grade="8.8")
        assert spec.kind is JointKind.BOLTED_THROUGH
        assert spec.method is ConnectionMethod.BOLTED
        assert spec.min_pitch_mm == pytest.approx(25.0)        # 2.5d
        assert spec.min_edge_mm == pytest.approx(15.0)         # 1.5d
        assert spec.full_strength_edge_mm == pytest.approx(20.0)
        assert spec.standard == "BS 5950 / EN 1993-1-8 Table 3.3"

    def test_medium_clearance_hole_for_m10_is_d_plus_one(self):
        spec = BoltedThroughSpec(nominal_diameter_mm=10.0)
        assert spec.resolved_hole_diameter_mm == pytest.approx(11.0)

    def test_explicit_hole_diameter_overrides_the_fit_table(self):
        spec = BoltedThroughSpec(nominal_diameter_mm=10.0,
                                 hole_diameter_mm=10.5)
        assert spec.resolved_hole_diameter_mm == pytest.approx(10.5)

    def test_hex_head_table_matches_iso_4014_for_common_sizes(self):
        assert hex_head_dimensions(10.0) == pytest.approx((16.0, 6.4))   # s, k
        assert hex_head_dimensions(12.0) == pytest.approx((18.0, 7.5))
        assert hex_head_dimensions(16.0) == pytest.approx((24.0, 10.0))

    def test_min_fillet_leg_follows_formula_and_aws_floor(self):
        assert min_fillet_leg_mm(10.0) == pytest.approx(5.0)   # 1.5*sqrt(10)=4.74 -> 5
        assert min_fillet_leg_mm(6.0) == pytest.approx(4.0)    # 3.67 -> series 4.0
        assert min_fillet_leg_mm(25.0) == pytest.approx(8.0)   # 7.5 -> AWS floor 8
        assert min_fillet_leg_mm(2.0) == pytest.approx(3.0)    # AWS floor 3

    def test_it7_bore_limits_for_10mm_h7_hole(self):
        lower, upper = it7_tolerance_mm(10.0)
        assert lower == pytest.approx(10.0)
        assert upper == pytest.approx(10.015)  # IT7 = 15 µm for 6-10 mm

    def test_interference_fit_accepts_only_iso286_classes(self):
        assert InterferenceSpec(nominal_diameter_mm=20.0, fit="H7/r6").kind \
            is JointKind.INTERFERENCE
        with pytest.raises(ValueError):
            InterferenceSpec(nominal_diameter_mm=20.0, fit="loose")

    def test_every_spec_kind_round_trips_through_its_registry(self):
        from connverify.joint_types import spec_from_dict, spec_to_dict
        specs = [
            BoltedThroughSpec(nominal_diameter_mm=10.0),
            BoltedTappedSpec(nominal_diameter_mm=8.0, thread_depth_mm=16.0),
            PinnedSpec(pin_diameter_mm=6.0),
            KeyedSpec(key_width_mm=8.0, key_height_mm=7.0),
            InterferenceSpec(nominal_diameter_mm=20.0, fit="H7/s6"),
            WeldedFilletSpec(design_leg_mm=5.0),
            RivetedSpec(rivet_diameter_mm=5.0),
            AdhesiveSpec(),
            ClampedSpec(),
            BearingSeatSpec(bore_diameter_mm=52.0),
        ]
        for spec in specs:
            clone = spec_from_dict(spec_to_dict(spec))
            assert clone == spec, spec.kind


class TestDslIntegration:
    def test_spec_implies_the_connection_method(self):
        iface = Interface(name="mount_face",
                          spec=BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert iface.method is ConnectionMethod.BOLTED

    def test_method_and_spec_together_is_an_error(self):
        with pytest.raises(TypeError):
            Interface(name="mount_face", method=ConnectionMethod.BOLTED,
                      spec=BoltedThroughSpec(nominal_diameter_mm=10.0))

    def test_env_with_spec_interfaces_validates_and_serializes(self):
        env = VerificationEnv(
            name="spec env",
            part_package="out/p.scadpkg",
            material=make_material(),
            interfaces=[
                Interface(name="mount_face",
                          spec=BoltedThroughSpec(nominal_diameter_mm=10.0,
                                                 bolt_grade="10.9")),
                Interface(name="load_pad", spec=ContactPadSpec_()),
            ],
            load_cases=[LoadCase(name="op", loads=[
                ForceLoad(target="load_pad", fz_n=-100.0)])],
        )
        env.validate()
        clone = VerificationEnv.from_json(env.to_json())
        assert clone.interfaces[0].spec == env.interfaces[0].spec
        assert clone.interfaces[0].spec.bolt_grade == "10.9"

    def test_unknown_spec_kind_in_json_is_rejected(self):
        env = _spec_env()
        import json as _json
        payload = _json.loads(env.to_json())
        payload["interfaces"][0]["spec"]["kind"] = "magnetic"
        with pytest.raises(EnvValidationError):
            VerificationEnv.from_json(_json.dumps(payload))


def ContactPadSpec_():
    from connverify.joint_types import ContactPadSpec
    return ContactPadSpec()


def _spec_env() -> VerificationEnv:
    return VerificationEnv(
        name="spec env",
        part_package="out/p.scadpkg",
        material=make_material(),
        interfaces=[
            Interface(name="mount_face",
                      spec=BoltedThroughSpec(nominal_diameter_mm=10.0)),
            Interface(name="load_pad", spec=ContactPadSpec_()),
        ],
        load_cases=[LoadCase(name="op", loads=[
            ForceLoad(target="load_pad", fz_n=-100.0)])],
    )
