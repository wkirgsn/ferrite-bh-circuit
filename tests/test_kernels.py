"""Fused Triton kernels of the recurrence (src/scan.py: constant coefficients, src/scan_tv.py: per-step coefficients)
against their plain PyTorch references, forward and hand-derived backward, on random inputs."""
import pytest
import torch
from conftest import gpu


def _inputs(tv, E=2, N=40, K=7, W=200, seed=0):
    """Random inputs of the scan: positional arguments and the keywords gamma, m (per step for scan_tv)."""
    torch.manual_seed(seed); dev = "cuda"
    mk = lambda *sh, sc=1.0: (torch.randn(*sh, device=dev) * sc).requires_grad_()
    pos = lambda *sh, sc=1.0: (torch.rand(*sh, device=dev) * sc).requires_grad_()
    co = (E, N, W, K) if tv else (E, 1, K)
    args = [mk(E, N, W, sc=.2), mk(E, N, W), mk(E, N, W), mk(E, N, K, sc=.2), mk(E, N), pos(*co),
            pos(*co, sc=.1), mk(*((E, N, W, K) if tv else (E, N, K)), sc=.3), pos(E, 1, K, sc=.05), pos(E, 1, sc=.3)]
    kw = dict(gamma=pos(*((E, N, W, K) if tv else (E, 1, K)), sc=.05), m=(1 + 0.3 * torch.rand(E, N, W, device=dev)).requires_grad_())
    return args, kw


@gpu
@pytest.mark.parametrize("module", ["scan", "scan_tv"])
def test_fused_equals_reference(module):
    """Output and every gradient of the fused kernel equal the reference (relative 1e-4)."""
    mod = __import__(module)
    args, kw = _inputs(module == "scan_tv")
    k = 80
    leaves = args + [kw["gamma"], kw["m"]]
    res = []
    for f in (mod.scan, mod.scan_ref):
        for a in leaves: a.grad = None
        h = f(*args, k, **kw)
        (h.sin() * torch.linspace(0, 1, h.shape[-1], device=h.device)).sum().backward()
        res.append([h.detach().clone()] + [a.grad.clone() if a.grad is not None else None for a in leaves])
    for i, (x, y) in enumerate(zip(*res)):
        if i == 3:                                                    # measured H: no gradient needed (data)
            continue
        assert x is not None and y is not None, i
        assert ((x - y).abs().max() / y.abs().max().clamp(min=1e-6)).item() < 1e-4, i
