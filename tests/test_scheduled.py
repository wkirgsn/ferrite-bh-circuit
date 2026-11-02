"""Gain-scheduled circuit (src/hyper_m.py): with the scheduling MLP switched off it is exactly the classical circuit
it starts from, and its parameter count matches the paper (656 / 5552 per material)."""
import pytest
import torch
from conftest import excitation
from mc2lib import ROOT
from compact_m import CompactM, DEV
from loader import load
import ferrite_bh as fb


@pytest.mark.parametrize("ckpt,npar", [("hyAB16j_long", 656), ("hyAB64j_s0", 5552)])
def test_parameter_count(ckpt, npar):
    m, names, mats = load(ckpt)
    assert m.n_params()[0] // len(mats) == npar


def test_zero_schedule_is_backbone():
    """HyperM with the shifts switched off reproduces CompactM with the same 32 parameters."""
    m, names, mats = load("hyAB64j_s0"); m.eval()
    ck = torch.load(ROOT / "models/hyAB64j_s0.pt", map_location=DEV)
    ref = CompactM([ck["variant"]], len(mats), ck["M"], ck["J"]).to(DEV)
    ref.load_state_dict({k[3:]: v for k, v in m.state_dict().items() if k.startswith("bb.")}, strict=False)
    B, T = excitation(N=4, seed=5)
    H = fb.simulate("N87", B, T, material_only=False)[:, -1000:]
    B = torch.tensor(B[:, -1000:], dtype=torch.float32, device=DEV); H = torch.tensor(H, dtype=torch.float32, device=DEV)
    T = torch.tensor(T, dtype=torch.float32, device=DEV); mid = torch.full((4,), mats.index("N87"), device=DEV)
    with torch.no_grad():
        m.zero = True; a = m(B, H, T, mid, 500); m.zero = False
        b = ref(B, H, T, mid, 500)
        c = m(B, H, T, mid, 500)
    scale = H.pow(2).mean().sqrt()
    assert ((a - b).abs().max() / scale).item() < 1e-5
    assert ((c - b).abs().max() / scale).item() > 1e-3                 # the trained schedule does change the output
