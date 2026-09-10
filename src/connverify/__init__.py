"""connverify — static connector verification environment addon for SimpleCADAPI.

Consumes `.scadpkg` product packages, verifies connection interfaces, load
cases and spatial envelopes, and runs linear-static FEM via FEMaster.
Units: millimeters, newtons, megapascals (N-mm-MPa).
"""

from .env import (
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
from .frd import FrdResult, parse_frd
from .joint_geometry import (
    HoleInfo,
    detect_holes,
    hole_axis_intersections,
    measure_local_thickness_mm,
    min_edge_distance_mm,
    pairwise_pitch_mm,
)
from .counterpart import (
    BoreCounterpart,
    PlateCounterpart,
    PlaneCounterpart,
    SnapCounterpart,
    counterpart_from_dict,
    counterpart_to_dict,
)
from .joint_types import (
    AdhesiveSpec,
    BearingSeatSpec,
    BoltedTappedSpec,
    BoltedThroughSpec,
    ClampedSpec,
    ContactPadSpec,
    InterferenceSpec,
    JointKind,
    KeyedSpec,
    PinnedSpec,
    RivetedSpec,
    SnapFitSpec,
    SplinedSpec,
    StudSpec,
    TransitionSpec,
    WeldedButtSpec,
    WeldedFilletSpec,
    hex_head_dimensions,
    it7_tolerance_mm,
    min_fillet_leg_mm,
)
from .package_reader import (
    InterfaceFace,
    InterfaceInfo,
    LoadedPart,
    MissingInterfaceError,
    UnsupportedPackageError,
    load_part,
    require_interfaces,
)
from .pipeline import verify
from .render import render_tag_review
from .report import CaseOutcome, VerificationReport
from .solver import SolveResult, SolverError, SolverNotFoundError, run_solve

__version__ = "0.4.0"

__all__ = [
    "AdhesiveSpec",
    "BearingSeatSpec",
    "BoltedTappedSpec",
    "BoltedThroughSpec",
    "ClampedSpec",
    "ContactPadSpec",
    "InterferenceSpec",
    "JointKind",
    "KeyedSpec",
    "PinnedSpec",
    "RivetedSpec",
    "SplinedSpec",
    "StudSpec",
    "SnapFitSpec",
    "TransitionSpec",
    "WeldedButtSpec",
    "WeldedFilletSpec",
    "HoleInfo",
    "detect_holes",
    "hex_head_dimensions",
    "it7_tolerance_mm",
    "min_fillet_leg_mm",
    "ConnectionMethod",
    "EnvValidationError",
    "ForceLoad",
    "Interface",
    "KeepOutBox",
    "LoadCase",
    "Material",
    "MomentLoad",
    "PressureLoad",
    "VerificationEnv",
    "load_part",
    "require_interfaces",
    "LoadedPart",
    "InterfaceInfo",
    "InterfaceFace",
    "MissingInterfaceError",
    "UnsupportedPackageError",
    "parse_frd",
    "FrdResult",
    "verify",
    "render_tag_review",
    "BoreCounterpart",
    "PlateCounterpart",
    "PlaneCounterpart",
    "SnapCounterpart",
    "VerificationReport",
    "CaseOutcome",
    "run_solve",
    "SolveResult",
    "SolverError",
    "SolverNotFoundError",
    "__version__",
]
