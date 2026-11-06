"""ferrite_bh: the 32-parameter B-H model of power ferrites (Kirchgaessner 2026) in plain NumPy, with fitted parameters
for 15 MagNet Challenge 2 materials and generators for SPICE subcircuits. See AGENTS.md at the repository root.
The trained PyTorch circuits (classical, MLP16- and MLP64-scheduled) are available as ferrite_bh.Circuit (needs PyTorch;
imported on first use, so the NumPy part does not)."""
from .model import materials, load, coefficients, predict, simulate, drive, DT, OutOfTrainingRangeWarning, check_range

__all__ = ["materials", "load", "coefficients", "predict", "simulate", "drive", "DT", "OutOfTrainingRangeWarning", "check_range", "Circuit", "seeds"]


def __getattr__(name):
    """Circuit and seeds from ferrite_bh.circuit, imported on first access (PyTorch only then)."""
    if name in ("Circuit", "seeds"):
        from . import circuit
        return getattr(circuit, name)
    raise AttributeError(f"module 'ferrite_bh' has no attribute {name!r}")
