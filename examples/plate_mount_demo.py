"""End-to-end demo: natural language -> formal env -> verify -> report.

Requirement (natural language):
  "A steel mounting plate, bolted to a wall along one side face. The motor
   foot presses down on its top face with 1 kN. The plate must not extend
   into the motor housing zone above it. Safety factor 1.5 on S355 steel."

Run:  .venv/bin/python examples/plate_mount_demo.py
"""

import simplecadapi as scad
from simplecadapi import capture

from connverify.env import (
    ConnectionMethod, ForceLoad, Interface, KeepOutBox, LoadCase, Material,
    VerificationEnv,
)
from connverify.pipeline import verify


# ---- 1. model the part (SimpleCADAPI) and tag the connection end faces ----
@scad.part(id="mount_plate")
def build_mount_plate() -> scad.Part:
    solid = scad.make_box_rsolid(
        width=60.0, height=40.0, depth=30.0,
        bottom_face_center=(30.0, 20.0, 0.0),
        left_face_tag="interface.mount_face",   # bolted to the wall (y=0 face)
        top_face_tag="interface.load_pad",      # motor foot presses here
    )
    return scad.make_part_rpart(part_id="mount_plate", body=solid,
                                name="mount_plate")


capture(build_mount_plate(), "out/mount_plate.scadpkg", include_scene=False)

# ---- 2. formalize the verification environment -----------------------------
env = VerificationEnv(
    name="mount plate static",
    part_package="out/mount_plate.scadpkg",
    material=Material(name="steel_s355", youngs_modulus_mpa=210000.0,
                      poisson_ratio=0.3, yield_strength_mpa=355.0,
                      density_t_per_mm3=7.85e-9),
    interfaces=[
        Interface(name="mount_face", method=ConnectionMethod.BOLTED),
        Interface(name="load_pad", method=ConnectionMethod.CONTACT),
    ],
    load_cases=[
        LoadCase(name="operational",
                 loads=[ForceLoad(target="load_pad", fz_n=-1000.0)]),
    ],
    envelopes=[
        KeepOutBox(name="motor_housing_clearance",
                   min_corner_mm=(100.0, 100.0, 100.0),
                   max_corner_mm=(200.0, 200.0, 200.0)),
    ],
    safety_factor_required=1.5,
)
env.validate()

# ---- 3. verify (mesh -> deck -> FEMaster solve -> report) -------------------
report = verify(env, out_dir="out/verify", mesh_size_mm=8.0)
print(report.to_markdown())
print("report.json / report.md written under out/verify/")
