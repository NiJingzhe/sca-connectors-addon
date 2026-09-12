"""Executable spec for the verification-environment DSL.

All quantities carry their unit in the field name (mm, N, MPa, N·mm).
Constructors stay dumb; VerificationEnv.validate() is the single validator
and must report ALL violations at once — agents repair faster with a
complete diagnosis than with a fail-fast chain.
"""

import json

import pytest

from connverify.env import (
    ConnectionMethod,
    EnvValidationError,
    ForceLoad,
    Interface,
    KeepOutBox,
    LoadCase,
    Material,
    MomentLoad,
    PressureLoad,
    VerificationEnv,
)


def make_material(**over):
    kw = dict(
        name="steel_s355",
        youngs_modulus_mpa=210000.0,
        poisson_ratio=0.3,
        yield_strength_mpa=355.0,
        density_t_per_mm3=7.85e-9,
    )
    kw.update(over)
    return Material(**kw)


def make_env(**over):
    """Minimal valid environment: bracket bolted to a wall, contact pad loaded."""
    kw = dict(
        name="bracket_static",
        part_package="out/bracket.scadpkg",
        material=make_material(),
        interfaces=[
            Interface(name="mount_face", method=ConnectionMethod.BOLTED),
            Interface(name="load_pad", method=ConnectionMethod.CONTACT),
        ],
        load_cases=[
            LoadCase(
                name="operational",
                loads=[
                    ForceLoad(target="load_pad", fx_n=0.0, fy_n=-500.0, fz_n=0.0),
                    MomentLoad(target="load_pad", mx_nmm=0.0, my_nmm=0.0, mz_nmm=20000.0),
                ],
            )
        ],
        envelopes=[KeepOutBox(name="motor_housing", min_corner_mm=(-30.0, -30.0, 40.0),
                              max_corner_mm=(30.0, 30.0, 80.0))],
        safety_factor_required=1.5,
    )
    kw.update(over)
    return VerificationEnv(**kw)


class TestMinimalValidEnv:
    def test_valid_env_passes(self):
        make_env().validate()  # must not raise

    def test_contact_is_a_legal_load_entry_method(self):
        # mount_face constrains, load_pad is a free contact surface — valid.
        make_env().validate()


class TestInterfaceRules:
    def test_duplicate_interface_names_rejected(self):
        env = make_env(interfaces=[
            Interface(name="mount_face", method=ConnectionMethod.BOLTED),
            Interface(name="mount_face", method=ConnectionMethod.WELDED),
        ])
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        assert any("mount_face" in msg and "duplicate" in msg for _, msg in ei.value.errors)

    def test_interface_name_must_be_slug(self):
        for bad in ["Mount Face", "-lead", "träil", "a" * 65, ""]:
            env = make_env(interfaces=[
                Interface(name=bad, method=ConnectionMethod.BOLTED),
                Interface(name="load_pad", method=ConnectionMethod.CONTACT),
            ])
            with pytest.raises(EnvValidationError):
                env.validate()

    def test_required_interface_names_carry_the_interface_prefix(self):
        env = make_env()
        assert env.required_interface_names() == {
            "interface.mount_face",
            "interface.load_pad",
        }

    def test_method_must_be_a_known_connection_method(self):
        with pytest.raises(TypeError):
            Interface(name="x", method="bolted-but-wrong-type")


class TestLoadRules:
    def test_load_target_must_reference_declared_interface(self):
        env = make_env(load_cases=[LoadCase(
            name="bad", loads=[ForceLoad(target="nonexistent", fy_n=-1.0)])])
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        assert any("nonexistent" in msg for _, msg in ei.value.errors)

    def test_force_load_needs_exactly_one_of_target_or_point(self):
        both = LoadCase(name="both", loads=[
            ForceLoad(target="load_pad", point_mm=(1.0, 2.0, 3.0), fy_n=-1.0)])
        neither = LoadCase(name="neither", loads=[ForceLoad(fy_n=-1.0)])
        for lc in (both, neither):
            env = make_env(load_cases=[lc])
            with pytest.raises(EnvValidationError):
                env.validate()

    def test_moment_load_follows_same_target_rules(self):
        env = make_env(load_cases=[LoadCase(
            name="bad", loads=[MomentLoad(mx_nmm=1.0)])])
        with pytest.raises(EnvValidationError):
            env.validate()

    def test_pressure_targets_interface_only(self):
        env = make_env(load_cases=[LoadCase(
            name="p", loads=[PressureLoad(interface="load_pad", magnitude_mpa=2.0)])])
        env.validate()  # declared interface — fine
        env2 = make_env(load_cases=[LoadCase(
            name="p", loads=[PressureLoad(interface="ghost", magnitude_mpa=2.0)])])
        with pytest.raises(EnvValidationError):
            env2.validate()

    def test_empty_load_case_rejected(self):
        env = make_env(load_cases=[LoadCase(name="empty", loads=[])])
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        assert any("empty" in msg for _, msg in ei.value.errors)

    def test_env_with_no_load_cases_rejected(self):
        with pytest.raises(EnvValidationError):
            make_env(load_cases=[]).validate()

    def test_zero_load_rejected(self):
        # an all-zero force carries no information and usually masks an error
        env = make_env(load_cases=[LoadCase(
            name="zero", loads=[ForceLoad(target="load_pad", fx_n=0.0, fy_n=0.0, fz_n=0.0)])])
        with pytest.raises(EnvValidationError):
            env.validate()


class TestStaticDeterminacy:
    def test_only_contact_interfaces_leave_the_part_unconstrained(self):
        env = make_env(interfaces=[
            Interface(name="load_pad", method=ConnectionMethod.CONTACT),
            Interface(name="another_pad", method=ConnectionMethod.CONTACT),
        ])
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        assert any("constrain" in msg for _, msg in ei.value.errors)

    def test_fixed_bolted_welded_all_constrain(self):
        for method in (ConnectionMethod.FIXED, ConnectionMethod.BOLTED,
                       ConnectionMethod.WELDED):
            env = make_env(interfaces=[
                Interface(name="mount_face", method=method),
                Interface(name="load_pad", method=ConnectionMethod.CONTACT),
            ])
            env.validate()


class TestMaterialRules:
    @pytest.mark.parametrize("field,bad", [
        ("youngs_modulus_mpa", 0.0),
        ("youngs_modulus_mpa", -1.0),
        ("poisson_ratio", 0.0),
        ("poisson_ratio", 0.5),
        ("poisson_ratio", -0.2),
        ("yield_strength_mpa", 0.0),
        ("density_t_per_mm3", -1e-9),
    ])
    def test_bad_material_values_rejected(self, field, bad):
        env = make_env(material=make_material(**{field: bad}))
        with pytest.raises(EnvValidationError):
            env.validate()


class TestEnvelopeRules:
    def test_box_with_min_geq_max_rejected(self):
        env = make_env(envelopes=[KeepOutBox(
            name="bad", min_corner_mm=(10.0, 0.0, 0.0), max_corner_mm=(10.0, 5.0, 5.0))])
        with pytest.raises(EnvValidationError):
            env.validate()

    def test_envelopes_are_optional(self):
        make_env(envelopes=[]).validate()


class TestCollectAllDiagnostics:
    def test_multiple_errors_reported_in_one_raise(self):
        env = make_env(
            interfaces=[Interface(name="only_face", method=ConnectionMethod.CONTACT)],
            load_cases=[
                LoadCase(name="bad_target",
                         loads=[ForceLoad(target="ghost", fy_n=-1.0)]),
                LoadCase(name="empty", loads=[]),
            ],
        )
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        fields = [f for f, _ in ei.value.errors]
        assert len(ei.value.errors) >= 3
        assert any("constrain" in msg for _, msg in ei.value.errors)
        assert any("ghost" in msg for _, msg in ei.value.errors)
        assert any("empty" in msg for _, msg in ei.value.errors)


class TestSerialization:
    def test_json_round_trip_preserves_semantics(self):
        env = make_env()
        clone = VerificationEnv.from_json(env.to_json())
        assert clone == env
        clone.validate()

    def test_to_json_is_deterministic(self):
        a = make_env().to_json()
        b = make_env().to_json()
        assert a == b
        # also byte-stable across dict orderings of the same content
        assert json.loads(a) == json.loads(b)

    def test_json_carries_schema_version(self):
        payload = json.loads(make_env().to_json())
        assert payload["schema_version"] == "1.0"
        assert payload["units"] == {"length": "mm", "force": "N", "stress": "MPa"}

    def test_from_json_rejects_unknown_schema_version(self):
        payload = json.loads(make_env().to_json())
        payload["schema_version"] = "9.9"
        with pytest.raises(EnvValidationError):
            VerificationEnv.from_json(json.dumps(payload))

    def test_round_trip_preserves_point_loads(self):
        env = make_env(load_cases=[LoadCase(name="point", loads=[
            ForceLoad(point_mm=(10.0, 20.0, 30.0), fz_n=-100.0)])])
        clone = VerificationEnv.from_json(env.to_json())
        assert clone.load_cases[0].loads[0].point_mm == (10.0, 20.0, 30.0)


class TestDefaultsAreExplicit:
    def test_safety_factor_default_is_exposed_not_silent(self):
        # default exists but must round-trip visibly into the report contract
        env = make_env()
        env_kwargs_default = VerificationEnv(
            name="x", part_package="p.scadpkg", material=make_material(),
            interfaces=[Interface(name="m", method=ConnectionMethod.FIXED)],
            load_cases=[LoadCase(name="l", loads=[
                ForceLoad(target="m", fy_n=-1.0)])]).safety_factor_required
        assert env_kwargs_default == 1.5
        assert env.safety_factor_required == 1.5


class TestMeshStudy:
    def test_valid_study_round_trips(self):
        from connverify.env import MeshStudy
        env = make_env(mesh_study=MeshStudy(
            sizes_mm=(12.0, 8.0, 5.333), qoi_tolerance_pct=2.0))
        clone = VerificationEnv.from_json(env.to_json())
        assert clone.mesh_study is not None
        assert clone.mesh_study.sizes_mm == (12.0, 8.0, 5.333)
        assert clone.mesh_study.qoi_tolerance_pct == 2.0

    def test_sizes_are_canonicalized_coarse_to_fine(self):
        from connverify.env import MeshStudy
        study = MeshStudy(sizes_mm=(5.333, 12.0, 8.0))
        assert study.sizes_mm == (12.0, 8.0, 5.333)

    def test_env_without_study_stays_valid_and_omits_the_key(self):
        payload = json.loads(make_env().to_json())
        assert "mesh_study" not in payload
        clone = VerificationEnv.from_json(json.dumps(payload))
        assert clone.mesh_study is None

    def test_single_size_rejected(self):
        from connverify.env import MeshStudy
        env = make_env(mesh_study=MeshStudy(sizes_mm=(8.0,)))
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        assert any(f.startswith("mesh_study")
                   for f, _m in ei.value.errors)

    def test_duplicate_sizes_rejected(self):
        from connverify.env import MeshStudy
        env = make_env(mesh_study=MeshStudy(sizes_mm=(12.0, 8.0, 8.0)))
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        assert any("distinct" in m for _f, m in ei.value.errors)

    @pytest.mark.parametrize("bad", [0.0, -2.0, 30.0])
    def test_out_of_range_tolerance_rejected(self, bad):
        from connverify.env import MeshStudy
        env = make_env(mesh_study=MeshStudy(
            sizes_mm=(12.0, 8.0, 5.333), qoi_tolerance_pct=bad))
        with pytest.raises(EnvValidationError) as ei:
            env.validate()
        assert any("qoi_tolerance_pct" in f for f, _m in ei.value.errors)

    def test_from_json_rejects_wrong_study_type(self):
        payload = json.loads(make_env().to_json())
        payload["mesh_study"] = {"sizes_mm": "big", "qoi_tolerance_pct": 2.0}
        with pytest.raises(EnvValidationError):
            VerificationEnv.from_json(json.dumps(payload))
