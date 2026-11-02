"""Shared fixtures of the test suite: synthetic excitation (no measurement data needed) and GPU / ngspice skips.
Run from the repository root: uv run --group dev pytest"""
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton kernel)")


def excitation(N=4, W=1000, f=100e3, seed=0, lead=2000):
    """N windows of a sinusoidal flux density with a third harmonic and a phase offset, 0.05..0.25 T peak, at 16 MHz,
    preceded by `lead` samples ramped up from B = 0 (demagnetized start). Returns (B_full (N, lead + W), T (N,))."""
    rng = np.random.default_rng(seed)
    t = np.arange(lead + W) * 62.5e-9
    pk = rng.uniform(0.05, 0.25, (N, 1)); ph = rng.uniform(0, 2 * np.pi, (N, 1)); h3 = rng.uniform(-0.2, 0.2, (N, 1))
    ramp = np.minimum(np.arange(lead + W) / (lead / 2), 1.0)
    B = pk * ramp * (np.sin(2 * np.pi * f * t + ph) - np.sin(ph) + h3 * np.sin(6 * np.pi * f * t)) / (1 + abs(h3))
    T = rng.choice([25.0, 50.0, 70.0], N)
    return B, T


@pytest.fixture(scope="session")
def ngspice():
    """Path of the ngspice binary; skips the test if it is not installed."""
    from spice_check import NGSPICE
    if shutil.which(NGSPICE) is None and not Path(NGSPICE).exists():
        pytest.skip("ngspice not installed (set NGSPICE=/path/to/ngspice)")
    return NGSPICE
