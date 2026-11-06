"""PyTorch API of the trained circuits, with the call signature of ferrite_bh.predict (GPU if available, else CPU).
Needs PyTorch and the repository's model code (src/, on the path after `uv sync`).

    import ferrite_bh as fb
    c = fb.Circuit("mlp64")                               # "classical" (32 parameters), "mlp16" or "mlp64"; seed=0..
    H = c.predict("N87", B, H_warmup, T=25)               # MC2 setting, B: (W,) or (N, W) at 16 MHz

Wraps the checkpoints of models/: the classical circuit is variant final_e1 of nere.pt (seed 0) and ne_s<seed>.pt, the
scheduled circuits are hyAB16j_long.pt / hyAB16j_s<seed>.pt and hyAB64j_s<seed>.pt (src/loader.py loads any checkpoint
with all its variants)."""
import numpy as np
import torch
from mc2lib import ROOT
from compact_m import DEV
from loader import load
from .model import check_range, load as load_params

CKPT = {"classical": lambda s: ("nere" if s == 0 else f"ne_s{s}", "final_e1"),
        "mlp16": lambda s: ("hyAB16j_long" if s == 0 else f"hyAB16j_s{s}", "final_e1+hyperAB16j"),
        "mlp64": lambda s: (f"hyAB64j_s{s}", "final_e1+hyperAB64j")}


def seeds(circuit):
    """Training seeds with a checkpoint in models/."""
    return [s for s in range(20) if (ROOT / f"models/{CKPT[circuit](s)[0]}.pt").exists()]


class Circuit:
    """A trained circuit (classical, MLP16- or MLP64-scheduled) of one training seed, for all 15 materials."""
    def __init__(self, circuit="classical", seed=0):
        """circuit: "classical", "mlp16" or "mlp64" (MLP-scheduled with 16 / 64 hidden units); seed: training seed."""
        name, variant = CKPT[circuit](seed)
        self.model, variants, self.materials = load(name)
        self.model.eval()
        self.e = variants.index(variant)
        self.label = "classical circuit" if circuit == "classical" else f"{circuit.upper()}-scheduled circuit"

    @torch.no_grad()
    def predict(self, material, B, H_warmup, T, k=None):
        """H (A/m) for windows of B (T) at 16 MHz, given the measured H over the first k samples (warm-up).
        B: (W,) or (N, W); H_warmup: (k,) / (N, k) or full-length (only the first k samples are used); T: degC, scalar or
        (N,). NumPy arrays or tensors; returns H with the type and shape of B: the warm-up samples as measured, then the
        prediction (the field as measured by the MC2 test bench). Warns (OutOfTrainingRangeWarning) outside the
        training range of the material."""
        tensor = torch.is_tensor(B)
        npy = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else x
        check_range(load_params(material), npy(B), npy(T), self.label, stacklevel=4)   # 4: past torch.no_grad
        f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=DEV)
        B2, Hw = torch.atleast_2d(f(B)), torch.atleast_2d(f(H_warmup))
        N, W = B2.shape
        k = Hw.shape[1] if k is None else k
        H = torch.zeros(N, W, device=DEV); H[:, :k] = Hw[:, :k]
        T = torch.broadcast_to(f(T), (N,)).contiguous()
        mid = torch.full((N,), self.materials.index(material), device=DEV)
        out = self.model(B2, H, T, mid, k)[self.e]
        out = out[0] if B2.shape != np.shape(B) else out
        return out.to(B.device) if tensor else out.cpu().numpy()
