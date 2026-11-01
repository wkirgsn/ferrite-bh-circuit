"""ferrite_bh: the 32-parameter B-H model of power ferrites (Kirchgaessner 2026) in plain NumPy, with fitted parameters
for 15 MagNet Challenge 2 materials and generators for SPICE subcircuits. See AGENTS.md at the repository root."""
from .model import materials, load, coefficients, predict, simulate, drive, DT
