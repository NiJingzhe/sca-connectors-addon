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
from .report import CaseOutcome, VerificationReport
from .solver import SolveResult, SolverError, SolverNotFoundError, run_solve

__version__ = "0.1.0"

__all__ = [
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
    "VerificationReport",
    "CaseOutcome",
    "run_solve",
    "SolveResult",
    "SolverError",
    "SolverNotFoundError",
    "__version__",
]
