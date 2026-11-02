"""The NumPy package reproduces the trained PyTorch classical circuit (models/nere.pt, variant final_e1), and the
PyTorch API (src/circuit.py) wraps the same checkpoints; synthetic windows, no measurement data needed."""
import numpy as np
import pytest
import torch
import ferrite_bh as fb
from conftest import excitation


@pytest.fixture(scope="module")
def windows():
    B, T = excitation(N=6, seed=3)
    out = {}
    for mat in fb.materials():
        H = fb.simulate(mat, B, T, material_only=False)
        out[mat] = (B[:, -1000:], H[:, -1000:], T)
    return out


@pytest.fixture(scope="module")
def classical():
    from circuit import Circuit
    return Circuit("classical")


@pytest.mark.parametrize("k", [100, 500, 900])
def test_numpy_reproduces_torch(windows, classical, k):
    for mat, (B, H, T) in windows.items():
        Pn = fb.predict(mat, B, H[:, :k], T)
        Pt = classical.predict(mat, B, H[:, :k], T)
        rel = np.sqrt(((Pn - Pt)[:, k:] ** 2).mean(1) / (Pn[:, k:] ** 2).mean(1))
        assert rel.max() < 1e-5, (mat, rel.max())


def test_circuit_api_types_and_batching(windows):
    from circuit import Circuit, seeds
    B, H, T = windows["N87"]
    for name in ("classical", "mlp16", "mlp64"):
        assert 0 in seeds(name)
        c = Circuit(name)
        assert sorted(c.materials) == fb.materials()
        P = c.predict("N87", B, H[:, :500], T)
        assert isinstance(P, np.ndarray) and P.shape == B.shape
        Pt = c.predict("N87", torch.tensor(B), torch.tensor(H[:, :500]), torch.tensor(T))
        assert torch.is_tensor(Pt) and Pt.shape == B.shape
        p1 = c.predict("N87", B[2], H[2, :500], T[2])
        assert p1.shape == B[2].shape
        assert np.abs(p1 - P[2]).max() < 1e-4 * np.abs(P[2]).max()
