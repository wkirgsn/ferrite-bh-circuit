"""Fused GPU scan (Triton) for the viscous-play + observer recurrence used by the compact models.

Per row (one window of one model variant), K operator states p_i and an offset c:
    x    = u_j + gamma_i (h_{j-1} - q_{j-1}) - p_i         (gamma: feedback of the output H minus an offset q)
    p    = p + alpha * (relu(x - r) - relu(-x - r))       (r = 0, alpha = beta gives a linear relaxation)
    h    = s_j + m_j sum_i w_i p_i + c                     (m: per-step multiplier of the hysteresis part, default 1)
    j<k: e = H_j - h;  p += L (1 - ga act) e;  c += Lc e;  h = H_j    (observer during the known-H warm-up;
         act_i = operator i was dragged in this step, ga in {0,1} gates the correction of dragged operators)
Forward and backward each run the whole time loop in one kernel; the backward is hand-derived and
recomputes each step from the stored pre-step states.
"""
import torch, triton, triton.language as tl

KEEP = False      # keep the state trajectory of the last call in STATES (..., W, K): states before each step
STATES = None


@triton.jit
def _fwd(u, s, mm, Hh, q, p0, c0, al, rr, ww, LL, Lc, gm, h0, ga, out, Ps, Cs, R, W, k, K: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR); rm = rows < R
    ks = tl.arange(0, K); i2 = rows[:, None] * K + ks[None, :]; m2 = rm[:, None]
    p = tl.load(p0 + i2, mask=m2, other=0.0)
    a = tl.load(al + i2, mask=m2, other=0.0); r = tl.load(rr + i2, mask=m2, other=0.0)
    w = tl.load(ww + i2, mask=m2, other=0.0); L = tl.load(LL + i2, mask=m2, other=0.0)
    c = tl.load(c0 + rows, mask=rm, other=0.0); lc = tl.load(Lc + rows, mask=rm, other=0.0)
    gam = tl.load(gm + i2, mask=m2, other=0.0); hp = tl.load(h0 + rows, mask=rm, other=0.0)
    gat = tl.load(ga + rows, mask=rm, other=0.0)
    for j in range(W):
        tl.store(Ps + j * R * K + i2, p, mask=m2); tl.store(Cs + j * R + rows, c, mask=rm)
        uj = tl.load(u + j * R + rows, mask=rm, other=0.0); sj = tl.load(s + j * R + rows, mask=rm, other=0.0)
        hj = tl.load(Hh + j * R + rows, mask=rm, other=0.0); mj = tl.load(mm + j * R + rows, mask=rm, other=0.0)
        qp = tl.load(q + tl.maximum(j - 1, 0) * R + rows, mask=rm, other=0.0)
        x = uj[:, None] + gam * (hp - qp)[:, None] - p
        free = 1 - gat[:, None] * (tl.abs(x) > r).to(tl.float32)
        p = p + a * (tl.maximum(x - r, 0.0) - tl.maximum(-x - r, 0.0))
        h = sj + mj * tl.sum(w * p, 1) + c
        g = (j < k).to(tl.float32)
        e = (hj - h) * g
        p = p + L * free * e[:, None]; c = c + lc * e
        hp = g * hj + (1 - g) * h
        tl.store(out + j * R + rows, hp, mask=rm)


@triton.jit
def _bwd(go, u, s, mm, Hh, q, al, rr, ww, LL, Lc, gm, h0, ga, out, Ps, Cs, gu, gs, gmm, gq, gp0, gc0, gal, grr, gww, gLL, gLc, ggm, R, W, k,
         K: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR); rm = rows < R
    ks = tl.arange(0, K); i2 = rows[:, None] * K + ks[None, :]; m2 = rm[:, None]
    a = tl.load(al + i2, mask=m2, other=0.0); r = tl.load(rr + i2, mask=m2, other=0.0)
    w = tl.load(ww + i2, mask=m2, other=0.0); L = tl.load(LL + i2, mask=m2, other=0.0)
    lc = tl.load(Lc + rows, mask=rm, other=0.0); gam = tl.load(gm + i2, mask=m2, other=0.0)
    gat = tl.load(ga + rows, mask=rm, other=0.0)
    carry = tl.zeros((BR,), tl.float32); ggam = tl.zeros((BR, K), tl.float32)
    gp = tl.zeros((BR, K), tl.float32); gc = tl.zeros((BR,), tl.float32)
    ga = tl.zeros((BR, K), tl.float32); gr = tl.zeros((BR, K), tl.float32)
    gw = tl.zeros((BR, K), tl.float32); gL = tl.zeros((BR, K), tl.float32); glc = tl.zeros((BR,), tl.float32)
    for jj in range(W):
        j = W - 1 - jj
        p = tl.load(Ps + j * R * K + i2, mask=m2, other=0.0); c = tl.load(Cs + j * R + rows, mask=rm, other=0.0)
        uj = tl.load(u + j * R + rows, mask=rm, other=0.0); sj = tl.load(s + j * R + rows, mask=rm, other=0.0)
        hj = tl.load(Hh + j * R + rows, mask=rm, other=0.0); goj = tl.load(go + j * R + rows, mask=rm, other=0.0)
        mj = tl.load(mm + j * R + rows, mask=rm, other=0.0)
        hp = tl.load(out + (j - 1) * R + rows, mask=rm & (j > 0), other=0.0) + tl.load(h0 + rows, mask=rm & (j == 0), other=0.0)
        qp = tl.load(q + tl.maximum(j - 1, 0) * R + rows, mask=rm, other=0.0)
        x = uj[:, None] + gam * (hp - qp)[:, None] - p
        act = (tl.abs(x) > r).to(tl.float32)
        Lf = L * (1 - gat[:, None] * act)
        dz = tl.maximum(x - r, 0.0) - tl.maximum(-x - r, 0.0)
        p1 = p + a * dz
        wp = tl.sum(w * p1, 1)
        h = sj + mj * wp + c
        g = (j < k).to(tl.float32)
        e = (hj - h) * g
        ge = tl.sum(Lf * gp, 1) + lc * gc
        gL += gp * (1 - gat[:, None] * act) * e[:, None]; glc += gc * e
        gh = g * (-ge) + (1 - g) * (goj + carry)
        tl.store(gs + j * R + rows, gh, mask=rm); tl.store(gmm + j * R + rows, gh * wp, mask=rm)
        gp1 = gp + (gh * mj)[:, None] * w
        gw += (gh * mj)[:, None] * p1
        gc = gc + gh
        d = a * act
        gpd = gp1 * d
        tl.store(gu + j * R + rows, tl.sum(gpd, 1), mask=rm)
        carry = tl.sum(gam * gpd, 1); ggam += gpd * (hp - qp)[:, None]
        if j > 0:
            tl.store(gq + (j - 1) * R + rows, -carry, mask=rm)
        else:
            tl.store(gq + rows, tl.load(gq + rows, mask=rm, other=0.0) - carry, mask=rm)
        ga += gp1 * dz
        sx = tl.where(x > 0, 1.0, -1.0)
        gr += gp1 * (-a * sx * act)
        gp = gp1 * (1 - d)
    tl.store(gp0 + i2, gp, mask=m2); tl.store(gc0 + rows, gc, mask=rm)
    tl.store(gal + i2, ga, mask=m2); tl.store(grr + i2, gr, mask=m2); tl.store(gww + i2, gw, mask=m2)
    tl.store(gLL + i2, gL, mask=m2); tl.store(gLc + rows, glc, mask=rm); tl.store(ggm + i2, ggam, mask=m2)


BR = 64


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uT, sT, mT, HT, qT, p0, c0, al, rr, ww, LL, Lc, gm, h0, ga, k):
        W, R = uT.shape; K = p0.shape[1]
        out = torch.empty_like(uT); Ps = torch.empty(W, R, K, device=uT.device); Cs = torch.empty(W, R, device=uT.device)
        _fwd[(triton.cdiv(R, BR),)](uT, sT, mT, HT, qT, p0, c0, al, rr, ww, LL, Lc, gm, h0, ga, out, Ps, Cs, R, W, k, K=K, BR=BR)
        ctx.save_for_backward(uT, sT, mT, HT, qT, al, rr, ww, LL, Lc, gm, h0, ga, out, Ps, Cs); ctx.k = k
        _Scan.last = Ps if KEEP else None
        return out

    @staticmethod
    def backward(ctx, go):
        uT, sT, mT, HT, qT, al, rr, ww, LL, Lc, gm, h0, ga, out, Ps, Cs = ctx.saved_tensors
        W, R = uT.shape; K = al.shape[1]
        gu, gs, gmm, gq = torch.empty_like(uT), torch.empty_like(uT), torch.empty_like(uT), torch.zeros_like(uT)
        gp0, gal, grr, gww, gLL = (torch.empty_like(al) for _ in range(5))
        gc0, gLc, ggm = torch.empty_like(Lc), torch.empty_like(Lc), torch.empty_like(al)
        _bwd[(triton.cdiv(R, BR),)](go.contiguous(), uT, sT, mT, HT, qT, al, rr, ww, LL, Lc, gm, h0, ga, out, Ps, Cs, gu, gs, gmm, gq, gp0, gc0,
                                    gal, grr, gww, gLL, gLc, ggm, R, W, ctx.k, K=K, BR=BR)
        return gu, gs, gmm, None, gq, gp0, gc0, gal, grr, gww, gLL, gLc, ggm, None, None, None


def scan(u, s, H, p0, c0, alpha, r, w, L, Lc, k, gamma=None, q=None, gate=None, m=None):
    """u, s, H: (..., W); p0, alpha, r, w, L, gamma: (..., K) (broadcastable); c0, Lc: (...). Returns h (..., W).
    The feedback h_{-1} at j = 0 is H_0 (minus q_0); q (..., W) defaults to 0; gate (...) in {0,1} defaults to 0;
    m (..., W) multiplies the operator sum, defaults to 1.
    K is padded to a power of two internally (padded operators have alpha = w = 0)."""
    lead = torch.broadcast_shapes(u.shape[:-1], s.shape[:-1], H.shape[:-1], p0.shape[:-1], c0.shape, alpha.shape[:-1],
                                  r.shape[:-1], w.shape[:-1], L.shape[:-1], Lc.shape)
    W = u.shape[-1]; K = p0.shape[-1]; KP = max(2, triton.next_power_of_2(K))
    seq = lambda x: x.expand(*lead, W).reshape(-1, W).T.contiguous()
    op = lambda x: torch.nn.functional.pad(x.expand(*lead, K).reshape(-1, K), (0, KP - K)).contiguous()
    sc = lambda x: x.expand(lead).reshape(-1).contiguous()
    gamma = torch.zeros_like(p0) if gamma is None else gamma
    HT = seq(H).detach()
    q = torch.zeros_like(u) if q is None else q
    m = torch.ones_like(u) if m is None else m
    out = _Scan.apply(seq(u), seq(s), seq(m), HT, seq(q), op(p0), sc(c0), op(alpha), op(r), op(w), op(L), sc(Lc), op(gamma), HT[0].contiguous(), sc(torch.zeros_like(c0) if gate is None else gate).detach(), int(k))
    if KEEP:
        global STATES
        STATES = _Scan.last[..., :K].permute(1, 0, 2).reshape(*lead, W, K)
    return out.T.reshape(*lead, W)


def scan_ref(u, s, H, p0, c0, alpha, r, w, L, Lc, k, gamma=None, q=None, gate=None, m=None):
    """Plain torch reference (same semantics)."""
    gamma = torch.zeros_like(p0) if gamma is None else gamma
    q = torch.zeros_like(u) if q is None else q
    p, c = p0, c0[..., None]; out = []; hp = H[..., :1]; ps = []
    for j in range(u.shape[-1]):
        ps.append(p.expand(*u.shape[:-1], p.shape[-1]))
        x = u[..., j:j + 1] + gamma * (hp - q[..., max(j - 1, 0):max(j - 1, 0) + 1]) - p
        free = 1 - (0 if gate is None else gate[..., None]) * (x.abs() > r).float()
        p = p + alpha * (torch.relu(x - r) - torch.relu(-x - r))
        h = s[..., j:j + 1] + (1 if m is None else m[..., j:j + 1]) * (w * p).sum(-1, keepdim=True) + c
        if j < k:
            e = H[..., j:j + 1] - h
            p = p + L * free * e; c = c + Lc[..., None] * e
            h = H[..., j:j + 1].expand_as(h)
        out.append(h); hp = h
    if KEEP:
        global STATES
        STATES = torch.stack(ps, -2)
    return torch.cat(out, -1)


if __name__ == "__main__":
    import time
    torch.manual_seed(0); dev = "cuda"
    E, N, K, W, k = 3, 50, 7, 300, 120
    mk = lambda *sh, sc=1.0: (torch.randn(*sh, device=dev, dtype=torch.float32) * sc).requires_grad_()
    args = [mk(E, N, W, sc=.2), mk(E, N, W), mk(E, N, W), mk(E, N, K, sc=.2), mk(E, N), torch.rand(E, 1, K, device=dev).requires_grad_(),
            (torch.rand(E, 1, K, device=dev) * .1).requires_grad_(), mk(E, N, K, sc=.3), (torch.rand(E, 1, K, device=dev) * .05).requires_grad_(), (torch.rand(E, 1, device=dev) * .3).requires_grad_()]
    gam = (torch.rand(E, 1, K, device=dev) * .05).requires_grad_(); qq = mk(E, N, W, sc=3.0)
    mm = (1 + 0.3 * torch.rand(E, N, W, device=dev)).requires_grad_()
    for f in (scan, scan_ref):
        for a in args + [gam, qq, mm]: a.grad = None
        h = f(*args, k, gamma=gam, q=qq, gate=(torch.arange(N, device=dev) % 2).float().expand(E, N), m=mm); (h.sin() * torch.linspace(0, 1, W, device=dev)).sum().backward()
        res = [h.detach().clone()] + [a.grad.clone() if a.grad is not None else None for a in args + [gam, qq, mm]]
        if f is scan: R1 = res
    for i, (x, y) in enumerate(zip(R1, res)):
        print(i, "max abs diff", None if x is None else (x - y).abs().max().item(), "scale", None if y is None else y.abs().max().item())
    E, N, K, W = 8, 2048, 8, 600
    args = [mk(E, N, W, sc=.2), mk(E, N, W), mk(E, N, W), mk(E, N, K, sc=.2), mk(E, N), torch.rand(E, 1, K, device=dev).requires_grad_(),
            (torch.rand(E, 1, K, device=dev) * .1).requires_grad_(), mk(E, N, K, sc=.3), (torch.rand(E, 1, K, device=dev) * .05).requires_grad_(), (torch.rand(E, 1, device=dev) * .3).requires_grad_()]
    for i in range(4):
        torch.cuda.synchronize(); t0 = time.time(); scan(*args, 200).square().mean().backward(); torch.cuda.synchronize()
        print(f"fused E={E} N={N} W={W}: {time.time()-t0:.4f}s")
