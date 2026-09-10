"""Executable spec for the FEMaster deck generator.

Deck syntax facts verified against FEMaster v2.8.0 (examples + importer
source):
- ``*CLOAD`` values are PER NODE, repeated for every node of the target —
  the generator must pre-distribute loads and emit one row per node.
- ``*SUPPORT`` data rows: fixed DOFs get 0.0, free DOFs get NAN.
- solids have translational DOF only; rotational components stay NAN.
- one deck per load case.
"""

import pytest

from connverify.deck import (
    DeckError,
    UnsupportedLoadError,
    generate_deck,
    generate_decks,
)
from connverify.env import (
    ConnectionMethod,
    ForceLoad,
    Interface,
    LoadCase,
    Material,
    PressureLoad,
    MomentLoad,
    VerificationEnv,
)
from tests.conftest import build_probe_box  # noqa: F401  (ensures SDK import ok)
from tests.test_mesh_model import single_tet_mesh


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
        load_cases=[
            LoadCase(name="op", loads=[ForceLoad(target="load", fz_n=-300.0)]),
        ],
        envelopes=[],
    )
    kw.update(over)
    return VerificationEnv(**kw)


class TestDeckStructure:
    def test_deck_contains_the_full_command_skeleton(self):
        deck = generate_deck(single_tet_mesh(), make_env(), make_env().load_cases[0])
        for keyword in ("*MODEL", "*NODE", "*ELEMENT, TYPE=C3D4", "*NSET",
                        "*MATERIAL", "*ELASTIC, TYPE=ISOTROPIC", "*DENSITY",
                        "*SOLIDSECTION", "*SUPPORT", "*CLOAD",
                        "*LOADCASE, TYPE=LINEARSTATIC", "*SUPPORTS", "*LOADS",
                        "*SOLVER", "*CONSTRAINTMETHOD", "*END"):
            assert keyword in deck, f"missing {keyword}"

    def test_nodes_and_elements_are_written_verbatim(self):
        deck = generate_deck(single_tet_mesh(), make_env(), make_env().load_cases[0])
        assert "1, 0, 0, 0" in deck
        assert "2, 1, 0, 0" in deck
        assert "1, 1, 2, 3, 4" in deck

    def test_constraining_interfaces_become_supports_with_nan_rot_dofs(self):
        deck = generate_deck(single_tet_mesh(), make_env(), make_env().load_cases[0])
        assert "*SUPPORT, SUPPORT_COLLECTOR=BC_OP" in deck
        assert "IF_MOUNT, 0.0, 0.0, 0.0, NAN, NAN, NAN" in deck

    def test_material_values_flow_through(self):
        deck = generate_deck(single_tet_mesh(), make_env(), make_env().load_cases[0])
        assert "210000, 0.3" in deck
        assert "7.85e-09" in deck

    def test_generate_decks_yields_one_deck_per_case(self):
        env = make_env(load_cases=[
            LoadCase(name="a", loads=[ForceLoad(target="load", fz_n=-1.0)]),
            LoadCase(name="b", loads=[ForceLoad(target="load", fz_n=-2.0)]),
        ])
        decks = generate_decks(single_tet_mesh(), env)
        assert set(decks) == {"a", "b"}
        assert decks["a"] != decks["b"]


class TestLoadDistribution:
    def test_interface_force_splits_by_tributary_share(self):
        env = make_env()
        deck = generate_deck(single_tet_mesh(), env, env.load_cases[0])
        # load face nodes {1,2,4}, equal tributary 1/6 each: -300/3 = -100 per node
        for node in (1, 2, 4):
            assert f"{node}, 0, 0, -100, 0.0, 0.0, 0.0" in deck, deck
        # node 3 is not on the load face
        assert "3, 0, 0, -100" not in deck

    def test_resultant_force_is_preserved_exactly(self):
        env = make_env(load_cases=[LoadCase(
            name="asym",
            loads=[ForceLoad(target="mount", fx_n=10.0, fy_n=-20.0, fz_n=30.0)])])
        mesh = single_tet_mesh()
        deck = generate_deck(mesh, env, env.load_cases[0])
        sx = sy = sz = 0.0
        for line in deck.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 7 and parts[0].isdigit():
                sx += float(parts[1]); sy += float(parts[2]); sz += float(parts[3])
        assert sx == pytest.approx(10.0)
        assert sy == pytest.approx(-20.0)
        assert sz == pytest.approx(30.0)

    def test_pressure_becomes_inward_nodal_forces(self):
        env = make_env(load_cases=[LoadCase(
            name="press", loads=[PressureLoad(interface="mount", magnitude_mpa=6.0)])])
        deck = generate_deck(single_tet_mesh(), env, env.load_cases[0])
        # mount face: area 0.5, tributary 1/6 per node, outward n=(0,0,-1);
        # inward push => f_i = -p * trib * n = -6*(1/6)*(0,0,-1) = (0,0,+1)
        for node in (1, 2, 3):
            assert f"{node}, 0, 0, 1, 0.0, 0.0, 0.0" in deck

    def test_point_load_attaches_to_nearest_node(self):
        env = make_env(load_cases=[LoadCase(
            name="pt", loads=[ForceLoad(point_mm=(0.9, 0.05, 0.05), fx_n=5.0)])])
        deck = generate_deck(single_tet_mesh(), env, env.load_cases[0])
        assert "2, 5, 0, 0, 0.0, 0.0, 0.0" in deck

    def test_point_load_far_from_mesh_fails_with_distance(self):
        env = make_env(load_cases=[LoadCase(
            name="far", loads=[ForceLoad(point_mm=(50.0, 50.0, 50.0), fx_n=5.0)])])
        with pytest.raises(DeckError) as ei:
            generate_deck(single_tet_mesh(), env, env.load_cases[0],
                          point_attach_tol_mm=1.0)
        assert "50" in str(ei.value) or "distance" in str(ei.value).lower()


class TestUnsupportedPhysics:
    def test_moment_load_is_rejected_with_conversion_guidance(self):
        env = make_env(load_cases=[LoadCase(
            name="m", loads=[MomentLoad(target="load", mz_nmm=1000.0)])])
        with pytest.raises(UnsupportedLoadError) as ei:
            generate_deck(single_tet_mesh(), env, env.load_cases[0])
        msg = str(ei.value)
        assert "couple" in msg.lower() or "force" in msg.lower()


class TestDeterminismAndGuards:
    def test_same_inputs_give_identical_deck(self):
        env = make_env()
        a = generate_deck(single_tet_mesh(), env, env.load_cases[0])
        b = generate_deck(single_tet_mesh(), env, env.load_cases[0])
        assert a == b

    def test_interface_without_mesh_nodes_fails_loudly(self):
        from connverify.mesh_model import Mesh, Tet
        empty = Mesh(
            nodes={1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0),
                   4: (0.0, 0.0, 1.0)},
            tets=(Tet(element_id=1, node_ids=(1, 2, 3, 4)),),
            interface_faces={"mount": (), "load": ()},
        )
        with pytest.raises(DeckError) as ei:
            generate_deck(empty, make_env(), make_env().load_cases[0])
        assert "mount" in str(ei.value)
