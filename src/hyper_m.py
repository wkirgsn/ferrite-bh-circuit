"""Hyper-network on the circuit parameters (as the thermal neural network scales a lumped-parameter thermal network):
per-material MLPs map the excitation at every step to shifts of the final model's coefficients, which then pass through
the same constraint transforms as in the model (passive at every step). The backbone is the final 32-parameter model
(nere:final_e1 option set); its own forward is re-implemented here for that option set only.

Inputs per step n (exogenous, computed before the recurrence): u_n / 0.2 T, asinh(dB_n / 1 mT), asinh(d2B_n / 0.1 mT),
t = (T - 50) / 25. The MLP output is made even in (u, dB, d2B), z(x) = (f(x) + f(-x)) / 2, so that the model stays
odd-symmetric, H(-B) = -H(B). Shifts (last layer zero-initialized: starts at the backbone):
  option A (output side, scan.py):  a1 * e^z, a3 + 100 z, Bs + 0.05 z, b * e^z, hysteresis sum * e^z      (5 outputs)
  option B (dynamics, scan_tv.py):   logit alpha + z; log r_i + z + z' chi_i; play density d0 + z, d1 + z;
                                     logit beta_j + z_j; c_j raw + z_j; gamma + 1e-3 (z + z' chi_i)            (11 outputs)
  option C (feedback, scan_c.py):   the shifts of A and B, the MLP also sees asinh(H_(n-1) / 10 A/m) of the model's own
                                     output (measured H during the warm-up); evenness uses (-x, -H).
The warm-up estimator is the backbone's. Counted parameters per material: backbone (32) + MLP (NIN -> h -> h -> NOUT).
Usage: python hyper_m.py --init nere:final_e1 --opt A|B|AB|C --hidden 16 [--joint] --name <out>   -> models/<out>.pt"""
import argparse, os, time
import numpy as np, torch, torch.nn as nn
from mc2lib import *
from common import GPUSampler, sre_E, summarize
from compact_m import CompactM, evaluate_M, DEV
import scan_tv, scan_c

NIN = 4
NW = 20                                                                         # warm-up encoder: samples 0, 5, ..., 95
OUTS = {"A": ["a1", "a3", "bs", "bw", "m"], "B": ["al", "r0", "r1", "d0", "d1", "be1", "be2", "c1", "c2", "g0", "g1"]}
FINAL = {"fbop": 1, "k0": 1, "cubic": 1, "wsmooth": 1, "qest": 1, "lqk": 1, "qest2": 1, "rgeo": 1, "gsmooth": 1, "tlin": 1,
         "passive": 1, "hlag": 3}


IXC = {o: i for i, o in enumerate(OUTS["A"] + OUTS["B"])}
outs_b = lambda J: OUTS["B"][:5] + [f"be{j + 1}" for j in range(J)] + [f"c{j + 1}" for j in range(J)] + OUTS["B"][-2:]   # J relaxations


def _coef(pre_j, hp, uj, cst):
    """Option C coefficients at one step from the MLP pre-activations of the exogenous inputs and the previous H (N,)."""
    w1h, w2, bh2, w3, bh3, a1b, a3b, Bsb, lbw, al0, lr, x, xg, dw, braw, cz0, gam0 = cst
    a = torch.tanh(pre_j + w1h * torch.asinh(hp / 10.0)[None, :, None])
    a = torch.tanh(torch.einsum("snh,nhg->sng", a, w2) + bh2[None])
    z = (torch.einsum("snh,nho->sno", a, w3) + bh3[None]).mean(0)              # (N,NOUT)
    g = lambda o: z[:, IXC[o]:IXC[o] + 1]
    e = lambda v: torch.exp(torch.clamp(v, max=15.0))
    Bs = (Bsb + 0.05 * g("bs")).clamp(min=0.15); bw = torch.exp(lbw + g("bw"))
    hsat = 10.0 * (e((uj - Bs) / bw) - e((-uj - Bs) / bw))
    static = a1b * torch.exp(g("a1")) * uj + 100 * (a3b + g("a3")) * uj ** 3 + hsat
    alpha = torch.sigmoid(al0 + g("al")).expand(-1, x.shape[-1])
    r = torch.exp(lr + g("r0") + g("r1") * x)
    w = -nn.functional.softplus(-((dw[:, :1] + g("d0")) + (dw[:, 1:] + g("d1")) * x))
    beta = 0.5 * torch.sigmoid(braw + torch.cat([g("be1"), g("be2")], -1))
    cz = -nn.functional.softplus(cz0 + torch.cat([g("c1"), g("c2")], -1))
    gamma = gam0 + 1e-3 * (g("g0") + g("g1") * xg)
    return (static[:, 0], hsat[:, 0], torch.exp(g("m"))[:, 0], torch.cat([alpha, beta], -1),
            torch.cat([r, torch.zeros_like(beta)], -1), torch.cat([w, cz], -1), gamma, r, w, cz)


def _step(pre_j, uj, Hj, hp, p, c, q, qh, cst, Lp, Lc, obs: bool):
    """One step of the option-C recurrence (fused by torch.compile). Returns new states p, c and h, static, hsat."""
    st, hs, m, a, rr, wc, gm = _coef(pre_j, hp, uj, cst)[:7]
    st, m = st + q * (st - hs), m * (1 + qh)
    xx = uj + gm * hp[:, None] - p
    p = p + a * (torch.relu(xx - rr) - torch.relu(-xx - rr))
    h = st + m * (p * wc).sum(-1) + c
    if obs:
        er = Hj - h
        p = p + Lp * er[:, None]; c = c + Lc * er; h = Hj
    return p, c, h, st, hs


EAGER = bool(os.environ.get("HYPER_EAGER"))                                     # reference: step loop in torch (slow)


class HyperM(nn.Module):
    def __init__(s, variant, mats, M, J, hidden, opt="AB", wenc=0):
        super().__init__()
        assert {k: v for k, v in variant.items() if k not in ("name", "shared", "energy_w")} == FINAL, "backbone: final option set only"
        s.bb = CompactM([variant], len(mats), M, J); s.E, s.nm, s.hidden, s.M, s.J, s.opt = 1, len(mats), hidden, M, J, opt
        outs = {"A": OUTS["A"], "B": outs_b(J)}
        assert opt != "C" or J == 2, "option C: two relaxations (scan_c.py)"
        s.outs = [o for g in "AB" if g in opt or opt == "C" for o in outs[g]]
        nm, h, no, NIN = len(mats), hidden, len(s.outs), 5 if opt == "C" else 4
        u = lambda *sh, fan: nn.Parameter(torch.empty(*sh).uniform_(-fan ** -0.5, fan ** -0.5))
        s.w1, s.b1 = u(nm, NIN, h, fan=NIN), u(nm, h, fan=NIN)
        s.w2, s.b2 = u(nm, h, h, fan=h), u(nm, h, fan=h)
        s.w3, s.b3 = nn.Parameter(torch.zeros(nm, h, no)), nn.Parameter(torch.zeros(nm, no))
        s.register_buffer("sg", torch.tensor([-1.0, -1.0, -1.0, 1.0]), persistent=False)  # input signs of the mirrored branch
        s.last_q = None; s.zero = False; s.on_dev = False; s.wenc = wenc
        if wenc:                                                                # warm-up encoder (estimator): initial-state correction
            ne = 2 * NW + 4
            s.e1, s.c1 = u(nm, ne, wenc, fan=ne), u(nm, wenc, fan=ne)
            s.e2, s.c2 = u(nm, wenc, wenc, fan=wenc), u(nm, wenc, fan=wenc)
            s.e3, s.c3 = nn.Parameter(torch.zeros(nm, wenc, M + J)), nn.Parameter(torch.zeros(nm, M + J))

    def n_params(s):
        per = sum(p[0].numel() for p in (s.w1, s.b1, s.w2, s.b2, s.w3, s.b3))
        return [s.bb.n_params()[0] + s.nm * per]

    @staticmethod
    def preroll_states(u, static, mmod, alpha, r, w, beta, cz, gamma, p, zz, frac=0.9, reps=2, tol=0.05):
        """Parameter-free prior of the initial states: the circuit is driven open loop (its own H as feedback) over the
        window's own B, from the start to the first later instant with the same value and direction once the excursion
        has covered frac of the window's range, reps times. Windows without such an instant keep the given prior."""
        uu = u[0]; N, W = uu.shape; M = p.shape[-1]; ar = torch.arange(N, device=u.device); 
        d = torch.gradient(uu, dim=1)[0]; u0, d0 = uu[:, :1], d[:, :1]
        rng = (uu.amax(1, keepdim=True) - uu.amin(1, keepdim=True)).clamp(min=1e-6)
        full = (torch.cummax(uu, 1)[0] - torch.cummin(uu, 1)[0]) >= frac * rng
        cost = (uu - u0).abs() / rng + 0.5 * (d - d0).abs() / d.abs().amax(1, keepdim=True).clamp(min=1e-9)
        cost = torch.where(full & (torch.sign(d) == torch.sign(d0)), cost, torch.full_like(cost, float("inf")))
        cmin, ns = cost.min(1); ok = torch.isfinite(cmin) & (cmin < tol)
        Ls = torch.where(ok, ns, torch.zeros_like(ns)) + 1
        st = lambda x, j: x[0, ar, j] if x.dim() == 4 else x[0]
        pp, z = p[0].clone(), zz[0].clone(); hp = torch.zeros(N, device=u.device)
        for t in range(reps * int(Ls.max())):
            act = (t < reps * Ls)[:, None]; j = t % Ls
            v = uu[ar, j][:, None] + st(gamma, j) * hp[:, None]
            x = v[:, :M] - pp; rr = st(r, j)
            pn = pp + st(alpha, j) * (torch.relu(x - rr) - torch.relu(-x - rr)); zn = z + st(beta, j) * (v[:, M:] - z)
            hj = static[0, ar, j] + mmod[0, ar, j] * (st(w, j) * pn).sum(-1) + (st(cz, j) * zn).sum(-1)
            pp, z, hp = torch.where(act, pn, pp), torch.where(act, zn, z), torch.where(act[:, 0], hj, hp)
        HyperM.preroll_ok = ok.float().mean().item()
        return torch.where(ok[:, None], pp, p[0])[None], torch.where(ok[:, None], z, zz[0])[None]

    def n_enc(s):
        """Parameters per material of the warm-up encoder (estimator, not counted in n_params)."""
        return sum(p[0].numel() for p in (s.e1, s.c1, s.e2, s.c2, s.e3, s.c3)) if s.wenc else 0

    def encode(s, B, HE, t, mid):
        """Initial-state shifts (N, M+J) from the first 100 warm-up samples (B and de-lagged H every 5th sample, the
        level and rms), odd in (B, H): d(x) = (g(x) - g(Sx)) / 2."""
        b, h = B[:, :5 * NW:5], HE[:, :5 * NW:5]
        sc = h.pow(2).mean(1, keepdim=True).sqrt() + 1e-3
        dB0 = torch.sign(B[:, 5:6] - B[:, :1])
        x = torch.cat([(b - B[:, :1]) / 0.05, h / sc, B[:, :1] / 0.2, dB0, torch.log(sc / 10), t[:, None]], 1)
        sg = torch.ones(x.shape[1], device=x.device); sg[:-2] = -1
        g = lambda x: torch.einsum("nh,nho->no", torch.tanh(torch.einsum("nh,nhg->ng", torch.tanh(torch.einsum("ni,nih->nh", x, s.e1[mid])
                      + s.c1[mid]), s.e2[mid]) + s.c2[mid]), s.e3[mid]) + s.c3[mid]
        return (g(x) - g(x * sg)) / 2

    def hyper(s, u, d1, d2, t, mid):
        """Shifts z (N,W,NOUT), even in (u, dB, d2B)."""
        N, W = u.shape
        x = torch.stack([u / 0.2, torch.asinh(d1 / 1e-3), torch.asinh(d2 / 1e-4), t[:, None].expand(N, W)], -1)   # (N,W,NIN)
        sg = s.sg
        x = torch.stack([x, x * sg])                                            # (2,N,W,NIN)
        a = torch.tanh(torch.einsum("snwi,nih->snwh", x, s.w1[mid]) + s.b1[mid][None, :, None])
        a = torch.tanh(torch.einsum("snwh,nhg->snwg", a, s.w2[mid]) + s.b2[mid][None, :, None])
        z = torch.einsum("snwh,nho->snwo", a, s.w3[mid]) + s.b3[mid][None, :, None]
        return z.mean(0)

    def forward_c(s, u, d1, d2, t, HE, mid, k, base, L, hla, H0):
        """Option C: coefficients from the MLP with the previous output as input, step by step. u, d1, d2, HE (N,W)."""
        bb, M, J = s.bb, s.M, s.J
        P = lambda n: bb.P(n, mid)[0]
        N, W = u.shape
        a1b, a3b, Bsb, lbw, al0, lr, x, dw, cz0, gam0 = base
        xg = torch.cat([x, torch.zeros_like(x[..., :J])], -1)
        braw = P("beta")
        w1, w2, bh2, w3, bh3 = s.w1[mid], s.w2[mid], s.b2[mid], s.w3[mid], s.b3[mid]
        xin = torch.stack([u / 0.2, torch.asinh(d1 / 1e-3), torch.asinh(d2 / 1e-4), t[:, None].expand(N, W)], -1)
        sg = s.sg
        pre = torch.einsum("snwi,nih->snwh", torch.stack([xin, xin * sg]), w1[:, :4]) + s.b1[mid][None, :, None]
        w1h = torch.stack([w1[:, 4], -w1[:, 4]])                                # (2,N,h): the mirror sees -H
        cst = (w1h, w2, bh2, w3, bh3, a1b, a3b, Bsb, lbw, al0, lr, x, xg, dw, braw, cz0, gam0)
        coef = lambda j, hp: _coef(pre[:, :, j], hp, u[:, j:j + 1], cst)

        st0, _, _, _, _, _, _, r0, w0, cz0_ = coef(0, HE[:, 0])
        u0 = u[:, :1]
        p = u0 - r0 * torch.tanh(P("g2") * u0 / r0 + torch.sign(u[:, 5:6] - u0) * P("g"))
        zz = u0.expand(-1, J)
        e0 = HE[:, :1] - st0[:, None] - (w0 * p).sum(-1, keepdim=True) - (cz0_ * zz).sum(-1, keepdim=True)
        K0 = P("K0")
        p = p + 1e-3 * K0[:, :M] * e0; zz = zz + 1e-3 * K0[:, M:M + J] * e0
        c0 = (P("cpr") + K0[:, -1:] * e0)[:, 0]
        s0 = torch.cat([p, zz], -1); Lp, Lc = 1e-3 * L[:, :M + J], L[:, -1]

        zM = torch.zeros_like(lr); zJ = torch.zeros_like(braw)
        bvals = torch.cat([a1b, a3b, Bsb, lbw, al0, dw, torch.zeros_like(a1b)], -1)
        lanes = torch.stack([torch.cat([lr, zJ], -1), xg, xg, gam0, torch.cat([zM, braw], -1), torch.cat([zM, cz0], -1)])

        def run(Wn, kk, q=None, qh=None):
            q, qh = (torch.zeros_like(c0), torch.zeros_like(c0)) if q is None else (q, qh)
            if not EAGER:
                return scan_c.scan_c(pre[:, :, :Wn], w1h, w2, bh2, w3, bh3, s0, c0, q, qh, u[:, :Wn], HE[:, :Wn], Lp, Lc,
                                     bvals, lanes, kk, M)
            p, c, hp = s0, c0, HE[:, 0]; out, sts, hss = [], [], []
            for j in range(Wn):
                p, c, h, st, hs = _step(pre[:, :, j], u[:, j:j + 1], HE[:, j], hp, p, c, q, qh, cst, Lp, Lc, j < kk)
                out.append(h); sts.append(st); hss.append(hs); hp = h
            return torch.stack(out, -1), torch.stack(sts, -1), torch.stack(hss, -1)

        q = qh = None
        if k >= 50:                                                             # amplitude factors q, qh (qest2)
            Ho, static, hsat = run(k, 1)
            tt = torch.linspace(-1, 1, k - 1, device=u.device)
            dtr = lambda v: (lambda y: y - (y * tt).sum(-1, keepdim=True) / (tt * tt).sum() * tt)(v - v.mean(-1, keepdim=True))
            yd = dtr(HE[:, 1:k] - Ho[:, 1:])
            n0 = torch.exp(P("lq") + P("lqk") * float(np.log(k / 300))); Sh = (dtr(HE[:, 1:k]) ** 2).mean(-1, keepdim=True)
            h1 = dtr(static[:, 1:k] - hsat[:, 1:k]); h2 = dtr(Ho[:, 1:] - static[:, 1:k] - c0[:, None])
            s11, s22, s12 = ((a * b).sum(-1, keepdim=True) for a, b in ((h1, h1), (h2, h2), (h1, h2)))
            s11, s22 = s11 + n0 * Sh + 1e-6, s22 + n0 * Sh + 1e-6
            b1, b2 = (yd * h1).sum(-1, keepdim=True), (yd * h2).sum(-1, keepdim=True); det = s11 * s22 - s12 ** 2
            q = ((s22 * b1 - s12 * b2) / det).clamp(-0.5, 0.5); qh = ((s11 * b2 - s12 * b1) / det).clamp(-0.5, 0.5)
            s.last_q = torch.cat([q, qh], -1)[None]; q, qh = q[:, 0], qh[:, 0]
        out = run(W, k, q, qh)[0]
        return CompactM.lag(out[None], hla, H0)

    def forward(s, B, H, T, mid, k, dt=None):
        """Final model with per-step coefficient shifts. B, H (N,W), T (N,). Returns (1,N,W)."""
        bb, M, J = s.bb, s.M, s.J
        if not s.on_dev:                                                        # shared masks on the device once (graph capture)
            bb.shared = {n: bb.shared.get(n, torch.zeros(bb.E)).to(B.device) for n in bb.p}; s.on_dev = True
        P = lambda n: bb.P(n, mid)
        N, W = B.shape
        hla = 0.6 * torch.sigmoid(P("hl"))                                      # (1,N,1) lag pole (hlag = 3)
        HE = (H[None] - hla * torch.cat([H[:, :1], H[:, :-1]], -1)[None]) / (1 - hla)
        t = ((T - 50) / 25)[None, :, None]
        tp = torch.cat([torch.ones_like(t), t, torch.zeros_like(t)], -1)        # tlin: no t^2 terms
        d1 = torch.gradient(B, dim=1)[0]; d2 = torch.gradient(d1, dim=1)[0]
        tau = (P("tau") * tp[..., None, :2]).sum(-1)
        u = B[None] + tau[..., :1] * d1[None] + tau[..., 1:] * d2[None]         # (1,N,W)
        Z = torch.zeros(N, W, len(s.outs), device=B.device) if s.zero or s.opt == "C" else s.hyper(u[0], d1, d2, (T - 50) / 25, mid)
        z = {o: Z[..., i][None] for i, o in enumerate(s.outs)}                  # (1,N,W) each
        g = lambda o: z.get(o, torch.zeros_like(u))
        a1 = (P("a1") * tp).sum(-1, keepdim=True) * torch.exp(g("a1"))
        a3 = 100 * ((P("a3") * tp).sum(-1, keepdim=True) + g("a3"))
        Bs = ((P("bs") * tp).sum(-1, keepdim=True) + 0.05 * g("bs")).clamp(min=0.15)
        bw = torch.exp(P("log_bw")[..., :1] + g("bw"))
        e = lambda x: torch.exp(torch.clamp(x, max=15.0))
        hsat = 10.0 * (e((u - Bs) / bw) - e((-u - Bs) / bw))
        static = a1 * u + a3 * u ** 3 + hsat
        mmod = torch.exp(g("m")) * torch.ones_like(u)
        lr0 = P("log_r"); lr0 = lr0[..., :1] + (lr0[..., -1:] - lr0[..., :1]) * torch.linspace(0, 1, M, device=B.device)
        lr = lr0 + (P("rT") * tp[..., 1:]).sum(-1, keepdim=True)                # (1,N,M)
        x = (lr - lr.mean(-1, keepdim=True)) / (lr.std(-1, keepdim=True) + 1e-6)
        dw = (P("dw") * torch.stack([tp, tp], -2)).sum(-1)                      # (1,N,2)
        cz0 = (P("cz") * tp[..., None, :]).sum(-1)                              # (1,N,J)
        gop = P("gop"); gop = torch.cat([gop[..., :1] * x, gop[..., M:]], -1)
        gam = P("gam"); gamma = 1e-3 * ((gam[..., :1] + gam[..., 1:] * t) + gop)   # (1,N,M+J)
        al0 = P("alpha") + (P("aT") * tp[..., 1:]).sum(-1, keepdim=True)        # (1,N,1) logit
        if s.opt == "C":
            base = ((P("a1") * tp).sum(-1, keepdim=True)[0], (P("a3") * tp).sum(-1, keepdim=True)[0],
                    (P("bs") * tp).sum(-1, keepdim=True)[0], P("log_bw")[0, :, :1], al0[0], lr[0], x[0], dw[0], cz0[0], gamma[0])
            return s.forward_c(u[0], d1, d2, (T - 50) / 25, HE[0], mid, k, base, P("L")[0], hla, H[None, :, :1])
        if "B" in s.opt:                                                        # per-step operator coefficients (1,N,W,K)
            zs = lambda o: z[o][..., None]
            alpha = torch.sigmoid(al0[:, :, None] + zs("al"))
            r = torch.exp(lr[:, :, None] + zs("r0") + zs("r1") * x[:, :, None])
            w = -nn.functional.softplus(-((dw[..., None, :1] + zs("d0")) + (dw[..., None, 1:] + zs("d1")) * x[:, :, None]))
            beta = 0.5 * torch.sigmoid(P("beta")[:, :, None] + torch.cat([zs(f"be{j + 1}") for j in range(J)], -1))
            cz = -nn.functional.softplus(cz0[:, :, None] + torch.cat([zs(f"c{j + 1}") for j in range(J)], -1))
            xg = torch.cat([x, torch.zeros_like(x[..., :J])], -1)[:, :, None]
            gamma = gamma[:, :, None] + 1e-3 * (zs("g0") + zs("g1") * xg)
            r0, w0, cz0_ = r[:, :, 0], w[:, :, 0], cz[:, :, 0]
        else:
            alpha = torch.sigmoid(al0); r = torch.exp(lr)
            w = -nn.functional.softplus(-(dw[..., :1] + dw[..., 1:] * x)); beta = 0.5 * torch.sigmoid(P("beta"))
            cz = -nn.functional.softplus(cz0)
            r0, w0, cz0_ = r, w, cz
        u0 = u[..., :1]
        p = u0 - r0 * torch.tanh(P("g2") * u0 / r0 + torch.sign(u[..., 5:6] - u0) * P("g"))
        zz = u0.expand(1, -1, J)
        if getattr(s, "preroll", False) and "B" in s.opt:                       # diagnostics (amp_joint.py): pre-roll prior
            p, zz = s.preroll_states(u, static, mmod, alpha, r, w, beta, cz, gamma, p, zz, **getattr(s, "preroll_kw", {}))
        if s.wenc:                                                              # learned initial-state correction (estimator)
            d = s.encode(B, HE[0], (T - 50) / 25, mid)[None]
            p = p + 0.5 * r0 * torch.tanh(d[..., :M]); zz = zz + 0.05 * torch.tanh(d[..., M:])
        e0 = HE[..., :1] - static[..., :1] - (w0 * p).sum(-1, keepdim=True) - (cz0_ * zz).sum(-1, keepdim=True)
        K0 = P("K0")
        p = p + 1e-3 * K0[..., :M] * e0; zz = zz + 1e-3 * K0[..., M:M + J] * e0
        c = P("cpr") + K0[..., -1:] * e0
        L = P("L")
        if "B" in s.opt:
            al_k = torch.cat([alpha.expand(-1, -1, -1, M), beta], -1); r_k = torch.cat([r, torch.zeros_like(beta)], -1)
            w_k = torch.cat([w, cz], -1); sl = lambda v, Wn: v[..., :Wn, :]
        else:
            al_k = torch.cat([alpha.expand(-1, -1, M), beta], -1); r_k = torch.cat([r, torch.zeros_like(beta)], -1)
            w_k = torch.cat([w, cz], -1); sl = lambda v, Wn: v
        run = lambda Wn, st, mm, kk: scan_tv.scan(u[..., :Wn], st[..., :Wn], HE[..., :Wn], torch.cat([p, zz], -1), c[..., 0],
                                                 sl(al_k, Wn), sl(r_k, Wn), sl(w_k, Wn), 1e-3 * L[..., :M + J], L[..., -1], kk,
                                                 sl(gamma, Wn), None, None, m=mm[..., :Wn])
        run_p = lambda Wn, st, mm, kk, p0: scan_tv.scan(u[..., :Wn], st[..., :Wn], HE[..., :Wn], p0, c[..., 0], sl(al_k, Wn),
                                                       sl(r_k, Wn), sl(w_k, Wn), 1e-3 * L[..., :M + J], L[..., -1], kk,
                                                       sl(gamma, Wn), None, None, m=mm[..., :Wn])
        if k >= 50:                                                             # amplitude factors q, qh (qest2)
            Ho = run(k, static, mmod, 1)
            tt = torch.linspace(-1, 1, k - 1, device=B.device)
            dtr = lambda v: (lambda y: y - (y * tt).sum(-1, keepdim=True) / (tt * tt).sum() * tt)(v - v.mean(-1, keepdim=True))
            yd = dtr(HE[..., 1:k] - Ho[..., 1:])
            n0 = torch.exp(P("lq") + P("lqk") * float(np.log(k / 300))); Sh = (dtr(HE[..., 1:k]) ** 2).mean(-1, keepdim=True)
            h1 = dtr(static[..., 1:k] - hsat[..., 1:k]); h2 = dtr(Ho[..., 1:] - static[..., 1:k] - c)
            s11, s22, s12 = ((a * b).sum(-1, keepdim=True) for a, b in ((h1, h1), (h2, h2), (h1, h2)))
            s11, s22 = s11 + n0 * Sh + 1e-6, s22 + n0 * Sh + 1e-6
            b1, b2 = (yd * h1).sum(-1, keepdim=True), (yd * h2).sum(-1, keepdim=True); det = s11 * s22 - s12 ** 2
            if getattr(s, "q_prior", None) is not None:                         # diagnostics (amp_diag.py): prior mean
                b1, b2 = b1 + n0 * Sh * s.q_prior[..., :1], b2 + n0 * Sh * s.q_prior[..., 1:]
            q = ((s22 * b1 - s12 * b2) / det).clamp(-0.5, 0.5); qh = ((s11 * b2 - s12 * b1) / det).clamp(-0.5, 0.5)
            qj = getattr(s, "qjoint", None)
            if qj is not None:
                # diagnostics (amp_joint.py): joint linearized fit of q, qh and the initial states over the warm-up. The
                # sensitivities of the open-loop H to each initial state replace the trend removal; states scaled by their
                # prior width (plays r_i, relaxations 0.02 T), prior weight lam * Sh as for q; offset unpenalized
                K = M + J; p0 = torch.cat([p, zz], -1); sg = torch.cat([r0, torch.full_like(zz, 0.02)], -1)
                Hp = run_p(k, static, mmod, 1, p0 + 1e-3 * sg * torch.eye(K, device=B.device)[:, None, :])
                Jc = (Hp[..., 1:] - Ho[..., 1:]) / 1e-3                                  # (K,N,k-1) per unit prior width
                dm = (lambda v: v - v.mean(-1, keepdim=True)) if not qj.get("trend") else dtr
                X = torch.stack([dm(static[0, :, 1:k] - hsat[0, :, 1:k]), dm(Ho[0, :, 1:] - static[0, :, 1:k] - c[0])] +
                                [dm(Jc[i]) for i in range(K)], -1)                     # (N,k-1,2+K)
                A = X.transpose(-1, -2) @ X; yv = dm(HE[0, :, 1:k] - Ho[0, :, 1:])
                pen = torch.cat([(n0 * Sh)[0].expand(-1, 2), qj["lam"] * Sh[0].expand(-1, K)], -1)
                A = A + torch.diag_embed(pen) + 1e-6 * torch.eye(2 + K, device=B.device)
                sol = torch.linalg.solve(A, (X * yv[..., None]).sum(-2))              # (N,2+K)
                q, qh = sol[None, :, :1].clamp(-0.5, 0.5), sol[None, :, 1:2].clamp(-0.5, 0.5)
                if qj.get("states"):
                    dp = qj.get("gain", 1.0) * sol[None, :, 2:] * sg
                    p, zz = p + dp[..., :M], zz + dp[..., M:]
            if getattr(s, "q_override", None) is not None:                      # diagnostics (oracle_amp.py)
                q, qh = s.q_override[..., :1], s.q_override[..., 1:]
            s.last_q = torch.cat([q, qh], -1)
            static, mmod = static + q * (static - hsat), mmod * (1 + qh)
        out = run(W, static, mmod, k)
        return CompactM.lag(out, hla, H[None, :, :1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="<ckpt>:<variant> of the backbone (final option set)")
    ap.add_argument("--opt", default="AB", choices=["A", "B", "AB", "C"])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--joint", action="store_true", help="also fine-tune the backbone's 32 parameters and its estimator")
    ap.add_argument("--lr_bb", type=float, default=2e-3, help="peak learning rate of the backbone with --joint")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=256, help="windows per material per step")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--win", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ks", default="100,500,900")
    ap.add_argument("--energy_w", type=float, default=1.0)
    ap.add_argument("--qreg", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wenc", type=int, default=0, help="hidden units of a warm-up encoder for the initial state (0: none)")
    ap.add_argument("--qjoint", type=float, default=0.0, help="joint fit of the amplitude factors and the initial-state "
                    "sensitivities over the warm-up, prior weight of the states (0: trend-removed fit)")
    ap.add_argument("--check", action="store_true", help="only verify that the zero hyper-network reproduces the backbone")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    ckn, var = args.init.split(":")
    src = torch.load(ROOT / f"models/{ckn}.pt", map_location=DEV); e0 = src["names"].index(var); mats = src["mats"]
    variant = src["variants"][e0]
    m = HyperM(variant, mats, src["M"], src["J"], args.hidden, args.opt, args.wenc).to(DEV)
    if args.qjoint: m.qjoint = dict(lam=args.qjoint)
    m.bb.load_state_dict({k: v[e0:e0 + 1] for k, v in src["sd"].items() if k.startswith("p.")}, strict=False)
    if args.check:
        ref = CompactM([variant], len(mats), src["M"], src["J"]).to(DEV)
        ref.load_state_dict({k: v[e0:e0 + 1] for k, v in src["sd"].items() if k.startswith("p.")}, strict=False)
        for i, mt in enumerate(mats):
            w = windows("val", mt); B, H, T = (torch.tensor(w[x][:200], device=DEV) for x in "BHT")
            mi = torch.full((len(B),), i, device=DEV)
            with torch.no_grad():
                for k in (100, 500, 900):
                    a, b = m(B, H, T, mi, k), ref(B, H, T, mi, k)
                    print(f"{mt} k={k}: max |dH| / rms H = {((a - b).abs().max() / H.pow(2).mean().sqrt()).item():.2e}", flush=True)
        raise SystemExit
    if not args.joint:
        for p in m.bb.parameters():
            p.requires_grad_(False)
    samplers = [GPUSampler(split_mat(mt)["train"], DEV) for mt in mats]
    gen = torch.Generator(DEV).manual_seed(1 + 1000 * args.seed); rng = np.random.default_rng(1 + 1000 * args.seed)
    npar = m.n_params(); names = [f"{var}+hyper{args.opt}{args.hidden}" + ("j" if args.joint else "") + (f"w{args.wenc}" if args.wenc else "") + ("qj" if args.qjoint else "")]
    print(f"device {DEV}; materials {mats}; {names[0]}: {npar[0] // len(mats)} params per material (+ {m.n_enc()} warm-up encoder); outputs {m.outs}", flush=True)
    vws = [{k: v[np.random.default_rng(0).permutation(len(v))][:350] for k, v in windows("val", mt).items()} for mt in mats]
    for i, mt in enumerate(mats):
        summarize(evaluate_M(m, vws[i], i), names, [n // len(mats) for n in npar], f"val {mt} (backbone)")
    groups = [{"params": [p for n, p in m.named_parameters() if not n.startswith("bb.")], "lr": args.lr}]
    if args.joint:
        groups.append({"params": list(m.bb.parameters()), "lr": args.lr_bb})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for it in range(args.steps):
        bt = [smp(args.bs, args.win, gen) for smp in samplers]
        B, H, T = (torch.cat(x) for x in zip(*bt))
        mid = torch.cat([torch.full((len(b[0]),), i, device=DEV) for i, b in enumerate(bt)])
        pos = torch.cat([torch.arange(len(b[0]), device=DEV) for b in bt])
        k = int(rng.choice([int(x) for x in args.ks.split(",")]))
        opt.zero_grad(); lossE = 0
        for c in range(args.accum):
            sel = pos % args.accum == c; Bc, Hc, mc = B[sel], H[sel], mid[sel]
            Hp = m(Bc, Hc, T[sel], mc, k)
            loss = sre_E(Hp, Hc, k)                                             # (1,N)
            dB = torch.gradient(Bc, dim=1)[0][None]
            loss = loss + args.energy_w * ((Hp[..., k:] - Hc[None, :, k:]) * dB[..., k:]).sum(-1).abs() / (Hc[None] * dB).abs().sum(-1).clamp(min=1e-9)
            if m.last_q is not None:
                loss = loss + args.qreg * (m.last_q ** 2).mean(-1); m.last_q = None
            lc = torch.stack([loss[:, mc == i].mean(1) for i in range(len(mats))], 1) / args.accum
            lc.sum().backward(); lossE = lossE + lc.detach()
        for p in m.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(0.0, 0.0, 0.0)
        nn.utils.clip_grad_norm_([p for n, p in m.named_parameters() if not n.startswith("bb.")], 1.0)
        opt.step(); sched.step()
        if it % 250 == 0:
            print(f"it {it} loss " + "/".join(f"{100*x:.2f}" for x in lossE[0].tolist()) + f"  t {time.time()-t0:.0f}s", flush=True)
        if (it % 2000 == 0 and it > 0) or it == args.steps - 1:
            for i, mt in enumerate(mats):
                summarize(evaluate_M(m, vws[i], i), names, [n // len(mats) for n in npar], f"val {mt}")
            torch.save({"kind": "hyper", "args": vars(args), "variant": variant, "names": names, "mats": mats, "M": src["M"], "J": src["J"],
                        "hidden": args.hidden, "opt": args.opt, "wenc": args.wenc, "qjoint": args.qjoint, "sd": m.state_dict(), "init": args.init}, ROOT / f"models/{args.name}.pt")
