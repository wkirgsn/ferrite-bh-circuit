"""Fused GPU scan (Triton) as in scan.py, but with time-varying operator coefficients: alpha, r, w and gamma are given per
step, (..., W, K), as produced by the scheduling MLP from the excitation (hyper_m.py). Same recurrence:
    x    = u_j + gamma_ij h_{j-1} - p_i
    p    = p + alpha_ij * (relu(x - r_ij) - relu(-x - r_ij))
    h    = s_j + m_j sum_i w_ij p_i + c
    j<k: e = H_j - h;  p += L e;  c += Lc e;  h = H_j
The backward returns per-step gradients of alpha, r, w, gamma instead of their sums over the window.
"""
import torch, triton, triton.language as tl
from scan import scan as scan_const


@triton.jit
def _fwd(u, s, mm, Hh, p0, c0, al, rr, ww, LL, Lc, gm, h0, out, Ps, Cs, R, W, k, K: tl.constexpr, BR: tl.constexpr):
    """Forward kernel with per-step coefficients (W, R, K); stores the pre-step states Ps, Cs."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR); rm = rows < R
    ks = tl.arange(0, K); i2 = rows[:, None] * K + ks[None, :]; m2 = rm[:, None]
    p = tl.load(p0 + i2, mask=m2, other=0.0)
    L = tl.load(LL + i2, mask=m2, other=0.0)
    c = tl.load(c0 + rows, mask=rm, other=0.0); lc = tl.load(Lc + rows, mask=rm, other=0.0)
    hp = tl.load(h0 + rows, mask=rm, other=0.0)
    for j in range(W):
        tl.store(Ps + j * R * K + i2, p, mask=m2); tl.store(Cs + j * R + rows, c, mask=rm)
        a = tl.load(al + j * R * K + i2, mask=m2, other=0.0); r = tl.load(rr + j * R * K + i2, mask=m2, other=0.0)
        w = tl.load(ww + j * R * K + i2, mask=m2, other=0.0); gam = tl.load(gm + j * R * K + i2, mask=m2, other=0.0)
        uj = tl.load(u + j * R + rows, mask=rm, other=0.0); sj = tl.load(s + j * R + rows, mask=rm, other=0.0)
        hj = tl.load(Hh + j * R + rows, mask=rm, other=0.0); mj = tl.load(mm + j * R + rows, mask=rm, other=0.0)
        x = uj[:, None] + gam * hp[:, None] - p
        p = p + a * (tl.maximum(x - r, 0.0) - tl.maximum(-x - r, 0.0))
        h = sj + mj * tl.sum(w * p, 1) + c
        g = (j < k).to(tl.float32)  # ty: ignore[unresolved-attribute]  (Triton: a tensor in the kernel)
        e = (hj - h) * g
        p = p + L * e[:, None]; c = c + lc * e
        hp = g * hj + (1 - g) * h
        tl.store(out + j * R + rows, hp, mask=rm)


@triton.jit
def _bwd(go, u, s, mm, Hh, al, rr, ww, LL, Lc, gm, h0, out, Ps, Cs, gu, gs, gmm, gp0, gc0, gal, grr, gww, gLL, gLc, ggm, R, W, k,
         K: tl.constexpr, BR: tl.constexpr):
    """Backward kernel: reverse time loop, per-step gradients of alpha, r, w and gamma."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR); rm = rows < R
    ks = tl.arange(0, K); i2 = rows[:, None] * K + ks[None, :]; m2 = rm[:, None]
    L = tl.load(LL + i2, mask=m2, other=0.0)
    lc = tl.load(Lc + rows, mask=rm, other=0.0)
    carry = tl.zeros((BR,), tl.float32)
    gp = tl.zeros((BR, K), tl.float32); gc = tl.zeros((BR,), tl.float32)
    gL = tl.zeros((BR, K), tl.float32); glc = tl.zeros((BR,), tl.float32)
    for jj in range(W):
        j = W - 1 - jj
        a = tl.load(al + j * R * K + i2, mask=m2, other=0.0); r = tl.load(rr + j * R * K + i2, mask=m2, other=0.0)
        w = tl.load(ww + j * R * K + i2, mask=m2, other=0.0); gam = tl.load(gm + j * R * K + i2, mask=m2, other=0.0)
        p = tl.load(Ps + j * R * K + i2, mask=m2, other=0.0); c = tl.load(Cs + j * R + rows, mask=rm, other=0.0)
        uj = tl.load(u + j * R + rows, mask=rm, other=0.0); sj = tl.load(s + j * R + rows, mask=rm, other=0.0)
        hj = tl.load(Hh + j * R + rows, mask=rm, other=0.0); goj = tl.load(go + j * R + rows, mask=rm, other=0.0)
        mj = tl.load(mm + j * R + rows, mask=rm, other=0.0)
        hp = tl.load(out + (j - 1) * R + rows, mask=rm & (j > 0), other=0.0) + tl.load(h0 + rows, mask=rm & (j == 0), other=0.0)
        x = uj[:, None] + gam * hp[:, None] - p
        act = (tl.abs(x) > r).to(tl.float32)
        dz = tl.maximum(x - r, 0.0) - tl.maximum(-x - r, 0.0)
        p1 = p + a * dz
        wp = tl.sum(w * p1, 1)
        h = sj + mj * wp + c
        g = (j < k).to(tl.float32)
        e = (hj - h) * g
        ge = tl.sum(L * gp, 1) + lc * gc
        gL += gp * e[:, None]; glc += gc * e
        gh = g * (-ge) + (1 - g) * (goj + carry)
        tl.store(gs + j * R + rows, gh, mask=rm); tl.store(gmm + j * R + rows, gh * wp, mask=rm)
        gp1 = gp + (gh * mj)[:, None] * w
        tl.store(gww + j * R * K + i2, (gh * mj)[:, None] * p1, mask=m2)
        gc = gc + gh
        d = a * act
        gpd = gp1 * d
        tl.store(gu + j * R + rows, tl.sum(gpd, 1), mask=rm)
        carry = tl.sum(gam * gpd, 1)
        tl.store(ggm + j * R * K + i2, gpd * hp[:, None], mask=m2)
        tl.store(gal + j * R * K + i2, gp1 * dz, mask=m2)
        sx = tl.where(x > 0, 1.0, -1.0)
        tl.store(grr + j * R * K + i2, gp1 * (-a * sx * act), mask=m2)
        gp = gp1 * (1 - d)
    tl.store(gp0 + i2, gp, mask=m2); tl.store(gc0 + rows, gc, mask=rm)
    tl.store(gLL + i2, gL, mask=m2); tl.store(gLc + rows, glc, mask=rm)


BR = 64


class _ScanTV(torch.autograd.Function):
    """Autograd function of the fused scan with per-step coefficients."""
    @staticmethod
    def forward(ctx, uT, sT, mT, HT, p0, c0, al, rr, ww, LL, Lc, gm, h0, k):
        """Run the forward kernel and keep its inputs and the pre-step states for the backward."""
        W, R = uT.shape; K = p0.shape[1]
        out = torch.empty_like(uT); Ps = torch.empty(W, R, K, device=uT.device); Cs = torch.empty(W, R, device=uT.device)
        _fwd[(triton.cdiv(R, BR),)](uT, sT, mT, HT, p0, c0, al, rr, ww, LL, Lc, gm, h0, out, Ps, Cs, R, W, k, K=K, BR=BR)  # ty: ignore[invalid-argument-type]  (Triton constexpr)
        ctx.save_for_backward(uT, sT, mT, HT, al, rr, ww, LL, Lc, gm, h0, out, Ps, Cs); ctx.k = k
        return out

    @staticmethod
    def backward(ctx, go):  # ty: ignore[invalid-method-override]  (autograd: one gradient per input)
        """Gradients of all differentiable inputs (none for the measured H, h0 and k)."""
        uT, sT, mT, HT, al, rr, ww, LL, Lc, gm, h0, out, Ps, Cs = ctx.saved_tensors
        W, R = uT.shape; K = LL.shape[1]
        gu, gs, gmm = torch.empty_like(uT), torch.empty_like(uT), torch.empty_like(uT)
        gal, grr, gww, ggm = (torch.empty_like(al) for _ in range(4))
        gp0, gLL = torch.empty_like(LL), torch.empty_like(LL)
        gc0, gLc = torch.empty_like(Lc), torch.empty_like(Lc)
        _bwd[(triton.cdiv(R, BR),)](go.contiguous(), uT, sT, mT, HT, al, rr, ww, LL, Lc, gm, h0, out, Ps, Cs, gu, gs, gmm, gp0, gc0,
                                    gal, grr, gww, gLL, gLc, ggm, R, W, ctx.k, K=K, BR=BR)  # ty: ignore[invalid-argument-type]  (Triton constexpr)
        return gu, gs, gmm, None, gp0, gc0, gal, grr, gww, gLL, gLc, ggm, None, None


def scan(u, s, H, p0, c0, alpha, r, w, L, Lc, k, gamma=None, m=None):
    """As scan.scan, but alpha, r, w, gamma may be per step, (..., W, K); constant ones (..., K) are passed to scan.scan.
    u, s, H, m: (..., W); p0, L: (..., K); c0, Lc: (...)."""
    if all(x is None or x.dim() == p0.dim() for x in (alpha, r, w, gamma)):
        return scan_const(u, s, H, p0, c0, alpha, r, w, L, Lc, k, gamma, m)
    lead = torch.broadcast_shapes(u.shape[:-1], s.shape[:-1], H.shape[:-1], p0.shape[:-1], c0.shape, L.shape[:-1], Lc.shape)
    W = u.shape[-1]; K = p0.shape[-1]; KP = max(2, triton.next_power_of_2(K))
    seq = lambda x: x.expand(*lead, W).reshape(-1, W).T.contiguous()
    op = lambda x: torch.nn.functional.pad(x.expand(*lead, K).reshape(-1, K), (0, KP - K)).contiguous()
    tv = lambda x: torch.nn.functional.pad((x if x.dim() == p0.dim() + 1 else x[..., None, :]).expand(*lead, W, K)
                                           .reshape(-1, W, K).transpose(0, 1), (0, KP - K)).contiguous()   # (W,R,KP)
    sc = lambda x: x.expand(lead).reshape(-1).contiguous()
    gamma = torch.zeros_like(p0) if gamma is None else gamma
    HT = seq(H).detach()
    m = torch.ones_like(u) if m is None else m
    out = _ScanTV.apply(seq(u), seq(s), seq(m), HT, op(p0), sc(c0), tv(alpha), tv(r), tv(w), op(L), sc(Lc), tv(gamma),
                        HT[0].contiguous(), int(k))
    return out.T.reshape(*lead, W)


def scan_ref(u, s, H, p0, c0, alpha, r, w, L, Lc, k, gamma=None, m=None):
    """Plain torch reference with per-step alpha, r, w, gamma (..., W, K) (constant ones (..., K) broadcast)."""
    st = lambda x, j: x[..., j, :] if x.dim() == p0.dim() + 1 else x
    gamma = torch.zeros_like(p0) if gamma is None else gamma
    p, c = p0, c0[..., None]; out = []; hp = H[..., :1]
    for j in range(u.shape[-1]):
        x = u[..., j:j + 1] + st(gamma, j) * hp - p
        p = p + st(alpha, j) * (torch.relu(x - st(r, j)) - torch.relu(-x - st(r, j)))
        h = s[..., j:j + 1] + (1 if m is None else m[..., j:j + 1]) * (st(w, j) * p).sum(-1, keepdim=True) + c
        if j < k:
            e = H[..., j:j + 1] - h
            p = p + L * e; c = c + Lc[..., None] * e
            h = H[..., j:j + 1].expand_as(h)
        out.append(h); hp = h
    return torch.cat(out, -1)

