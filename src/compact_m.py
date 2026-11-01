"""Multi-material version of the compact law: parameters are grouped into ones shared by all materials
(instrument / measurement chain, or universal) and material-specific ones.

Law (one sample = 62.5 ns, t = (T - 50)/25, all coefficient polynomials in t):
    u   = B + tau1(T) B' + tau2(T) B''                             measurement low-pass undone
    v_i = u + gamma_i(T) (H_prev - [fbsat] H_sat(u_prev))          drive of operator i (H feedback)
    z_j <- z_j + beta_j (v_j - z_j)                                 linear relaxations
    p_i <- p_i + alpha(T) dz_{r_i(T)}(v_i - p_i)                    viscous play operators
    H   = a1(T) u + 10 A/m [e^{(u-Bs(T))/b} - e^{(-u-Bs(T))/b}] + sum w_i(T) p_i + sum c_j(T) z_j + c
Variants (leading axis E) differ in which parameter names are shared across materials ("shared" list) and in the
options cubic (a3 u^3 term), nofb (no H feedback), fbop (per-operator gamma offsets), fbsat (feed back H without the saturation term), bwT (knee width linear in T),
klin (knee exponential up to (u - Bs)/b = x0, linear beyond, x0 fitted: finite slope in deep saturation),
knee2 (second exponential knee 10 A/m e^{(|u| - Bs - d)/b2}, offset d and width b2 >= 5 mT fitted), knee3 (knee2 and a third
exponential knee of the same form), kpos (offsets of the further knees d = softplus(20 d') / 20 >= 0: beyond the main knee),
exc (excess-loss field 100 kex(T) sign(B') |B'|^(1/2), Bertotti),
qest (estimator: per-window amplitude factor 1 + q fitted on the warm-up), tlin (all coefficients at most linear in T),
rgeo (play widths log-equidistant between the first and last log_r), gsmooth (per-operator feedback offsets of the plays
linear in the standardized log r: gop_i = gop_0 x_i), lqk (prior strength of the amplitude estimate scales as k^lqk),
qest2 (separate amplitude factors of the reversible curve and of the operators), tauc (measurement correction
independent of T), revinit (non-causal warm-up estimator without observer gains, offset not fed back; see revinit()),
passive (equivalent circuit of positive elements apart from the feedback: w_i <= 0, c_j <= 0, beta_j <= 0.5),
ablations of single elements: noknee (no saturation knee), rateind (rate-independent plays, alpha = 1), norel (no
relaxations), notau (no measurement correction, u = B),
gamc, a3c, dwc (feedback gain, cubic coefficient, slope of the play density independent of T),
hlag (first-order low-pass of the H measurement, pole a per material: instrument; 1: warm-up de-lagged with the exact
inverse, 2: warm-up used as measured, only the prediction filtered, 3: as 1 with a <= 0.6), rvobs (candidate initial history + observer; 2: also q, qh from the candidate fit; 3: only q, qh from it).
Estimator parameters (g, g2, L, K0, cpr) are always per material and not counted. Estimator options: k0 (initial
mismatch spread over all states with gains K0, offset prior cpr, instead of c absorbing it), gate (observer does not
correct operators that are currently dragged by the drive).
"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn
from mc2lib import *
from common import GPUSampler, sre_E, summarize
from scan import scan

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu":    # no GPU: the plain torch reference of the fused kernel (same results, much slower)
    from scan import scan_ref as scan
if DEV.type == "cuda":   # the GPU also drives the desktop: leave the compositor ~2.5 GB of VRAM (a full GPU hung the login screen)
    torch.cuda.set_per_process_memory_fraction(0.84)
LAW = ["tau", "a1", "a3", "bs", "log_bw", "kx", "k2", "k3", "log_r", "rT", "alpha", "aT", "w", "beta", "cz", "gam", "gop", "kap", "dw", "kex", "hl"]


def init_params(M, J):
    """Base shape and initial value of every parameter (without the E and material axes)."""
    return {
        "tau": torch.tensor([[1.6, 0.0], [5.6, 0.0]]),          # [tau1, tau2] x [1, t]
        "a1": torch.tensor([180.0, 0.0, 0.0]),                    # [1, t, t^2]
        "a3": torch.zeros(3),                                     # cubic, x100 A/m/T^3 (option "cubic")
        "bs": torch.tensor([0.40, 0.0, 0.0]), "log_bw": torch.tensor([np.log(0.03), 0.0]),   # [1, t]
        "kx": torch.tensor([1.5]),                                # option klin: end x0 of the exponential part of the knee
        "k2": torch.tensor([0.05, np.log(0.03)]),                 # option knee2: offset d (T) and log width of the second knee
        "k3": torch.tensor([0.10, np.log(0.03)]),                 # option knee3: the same for the third knee
        "log_r": torch.linspace(np.log(0.01), np.log(0.25), M), "rT": torch.zeros(2),
        "alpha": torch.tensor([-1.5]), "aT": torch.zeros(2),
        "w": torch.cat([torch.full((M, 1), -10.0), torch.zeros(M, 2)], 1),
        "beta": torch.tensor([-4.2, -1.4, -0.5][:J]) if J <= 3 else torch.linspace(-5.5, -0.5, J), "cz": torch.zeros(J, 3),
        "gam": torch.zeros(2), "gop": torch.zeros(M + J),
        "kap": torch.zeros(2),                                    # option wmod: hysteresis amplitude exp(kap(T) u^2)
        "dw": torch.tensor([[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # option wsmooth: w_i = dw0(T) + dw1(T) x_i, x = std. log r_i
        "kex": torch.zeros(2),                                    # option exc: excess-loss field 100 kex(T) sign(B') |B'|^(1/2)
        "hl": torch.tensor([-3.0]),                               # option hlag: H-measurement low-pass pole a = 0.9 sigmoid(hl)
        "g": torch.zeros(M), "g2": torch.zeros(M), "L": torch.zeros(M + J + 1),
        "K0": torch.cat([torch.zeros(M + J), torch.ones(1)]), "cpr": torch.zeros(1),
        "lq": torch.tensor([float(np.log(300.0))]),               # option qest: log prior strength (samples) of the amplitude estimate
        "lqk": torch.zeros(1),                                    # option lqk: n0 = exp(lq) (k/300)^lqk
    }


class CompactM(nn.Module):
    def __init__(s, variants, n_mat, M=6, J=2):
        """variants: list of dicts {"shared": [names], "fbop", "fbsat", "bwT": 0/1}."""
        super().__init__()
        E = len(variants); s.E, s.M, s.J, s.nm = E, M, J, n_mat
        s.shared = {k: torch.tensor([float(k in v.get("shared", [])) for v in variants]) for k in LAW}
        for o in ("fbop", "fbsat", "bwT", "k0", "gate", "nofb", "cubic", "wlin", "bslin", "softbs", "wmod", "wsmooth", "exc", "qest", "tlin", "rgeo", "gsmooth", "lqk", "qest2", "tauc", "revinit", "passive", "gamc", "a3c", "dwc", "hlag", "rvobs", "noknee", "rateind", "norel", "notau", "klin", "knee2", "knee3", "kpos"):
            s.register_buffer(f"m_{o}", torch.tensor([float(v.get(o, 0)) for v in variants]))
        s.p = nn.ParameterDict({k: nn.Parameter(v.float().expand(E, n_mat, *v.shape).clone()) for k, v in init_params(M, J).items()})
        with torch.no_grad():                                                   # per-variant initial H-lag pole (hlinit)
            for e, v in enumerate(variants):
                if "hlinit" in v: s.p["hl"][e] = float(v["hlinit"])

    def n_params(s):
        out = []
        for e in range(s.E):
            n = 0
            for k in LAW:
                if k == "gop" and not s.m_fbop[e]: continue
                if k in ("gam", "gop") and s.m_nofb[e]: continue
                if k == "a3" and not s.m_cubic[e]: continue
                if k == "kap" and not s.m_wmod[e]: continue
                if k == "dw" and not s.m_wsmooth[e]: continue
                if k == "kex" and not s.m_exc[e]: continue
                if k == "hl" and not s.m_hlag[e]: continue
                if (k == "kx" and not s.m_klin[e]) or (k == "k2" and not (s.m_knee2[e] or s.m_knee3[e])) or (k == "k3" and not s.m_knee3[e]): continue
                if (k in ("bs", "log_bw") and s.m_noknee[e]) or (k in ("alpha", "aT") and s.m_rateind[e]) \
                        or (k in ("beta", "cz") and s.m_norel[e]) or (k == "tau" and s.m_notau[e]): continue
                if k == "w" and s.m_wsmooth[e]: continue
                size = 1 + int(s.m_bwT[e]) if k == "log_bw" else s.p[k][0, 0].numel()
                if k == "log_r" and s.m_rgeo[e]: size = 2
                if k == "tau" and s.m_tauc[e]: size = 2
                if k == "gop" and s.m_gsmooth[e]: size = 1 + s.J * (1 - int(s.m_norel[e]))
                if k in ("w", "cz") and s.m_wlin[e]: size = size // 3 * 2      # weights linear in T
                if k == "bs" and s.m_bslin[e]: size = 2
                if s.m_tlin[e]:                                                  # no t^2 coefficients
                    if k in ("a1", "a3", "rT", "aT") or (k == "bs" and not s.m_bslin[e]): size -= 1
                    if (k in ("w", "cz") and not s.m_wlin[e]) or k == "dw": size = size // 3 * 2
                if (k == "a3" and s.m_a3c[e]) or (k == "gam" and s.m_gamc[e]): size = 1
                if k == "dw" and s.m_dwc[e]: size -= 1 if s.m_tlin[e] else 2
                n += size * (1 if s.shared[k][e] else s.nm)
            out.append(n)
        return out

    def P(s, name, mid):
        """Parameter per (variant, row): (E, N, *shape); shared names use material slot 0."""
        sh = s.shared.get(name, torch.zeros(s.E)).to(mid.device)
        idx = (mid[None, :] * (1 - sh[:, None])).long()                       # (E,N)
        return s.p[name][torch.arange(s.E, device=mid.device)[:, None], idx]

    def drive(s, B, T, mid):
        """Measurement-corrected flux density u (E,N,W) at the native 16 MHz grid."""
        t = ((T - 50) / 25)[None, :, None]; tp = torch.cat([torch.ones_like(t).expand(s.E, -1, -1), t * (1 - s.m_tauc[:, None, None])], -1)
        d1 = torch.gradient(B, dim=1)[0]; d2 = torch.gradient(d1, dim=1)[0]
        tau = (s.P("tau", mid) * tp[..., None, :]).sum(-1)
        return B[None] + tau[..., :1] * d1[None] + tau[..., 1:] * d2[None]

    def forward(s, B, H, T, mid, k, dx0=None, gain=1.0, dt=None, u_in=None):
        """B, H (N,W), T (N,), material index mid (N,). Returns (E,N,W).
        dx0 (E,N,M+J): correction added to the prior initial states (moving-horizon estimation); gain scales L.
        dt (N,): step size in units of the native 62.5 ns (the law read as a discretized ODE: derivatives are divided by
        dt, the play and relaxation rates become 1 - (1 - alpha)^dt). The measurement correction is a filter of the 16 MHz
        acquisition and is not step-invariant: for dt != 1 pass u_in (E,N,W), computed with drive() at 16 MHz and resampled."""
        E, M, J = s.E, s.M, s.J
        P = lambda n: s.P(n, mid)
        HE = H[None]
        if s.m_hlag.any():
            # option hlag: the H channel of the instrument is a first-order low-pass y_n = a y_(n-1) + (1 - a) H_n of the
            # field H the law computes. The warm-up is de-lagged with the exact inverse; the output is filtered (see lag()).
            cap = torch.where(s.m_hlag == 3, 0.6, 0.9)[:, None, None]            # hlag = 3: as 1 with the pole capped at 0.6
            hla = (s.m_hlag > 0).float()[:, None, None] * cap * torch.sigmoid(P("hl"))       # (E,N,1)
            hli = hla * (s.m_hlag != 2).float()[:, None, None]                 # hlag = 2: warm-up not de-lagged
            HE = (H[None] - hli * torch.cat([H[:, :1], H[:, :-1]], -1)[None]) / (1 - hli)
        t = ((T - 50) / 25)[None, :, None]                                      # (1,N,1)
        tp = torch.cat([torch.ones_like(t), t, t ** 2], -1)                     # (1,N,3)
        tp = torch.cat([tp[..., :2].expand(E, -1, -1), tp[..., 2:] * (1 - s.m_tlin[:, None, None])], -1)   # option tlin: no t^2 terms
        d1 = torch.gradient(B, dim=1)[0]; d2 = torch.gradient(d1, dim=1)[0]
        if dt is not None:
            d1 = d1 / dt[:, None]; d2 = d2 / dt[:, None] ** 2
            rate = lambda a: 1 - (1 - a).clamp(min=1e-12) ** dt[None, :, None]
        tpt = torch.cat([tp[..., :1].expand(E, -1, -1), tp[..., 1:2] * (1 - s.m_tauc[:, None, None])], -1)   # option tauc
        tau = (P("tau") * tpt[..., None, :]).sum(-1) * (1 - s.m_notau[:, None, None])   # (E,N,2); notau: u = B
        u = B[None] + tau[..., :1] * d1[None] + tau[..., 1:] * d2[None] if u_in is None else u_in   # (E,N,W)
        a1 = (P("a1") * tp).sum(-1, keepdim=True)
        tpb = torch.cat([tp[..., :2].expand(E, -1, -1), tp[..., 2:] * (1 - s.m_bslin[:, None, None])], -1)  # option bslin
        Bs = (P("bs") * tpb).sum(-1, keepdim=True)
        sb = s.m_softbs[:, None, None]       # softbs: smooth floor at 0.02 T (the hard clamp at 0.15 T blocks the gradient)
        Bs = sb * (0.02 + nn.functional.softplus(50 * (Bs - 0.02)) / 50) + (1 - sb) * Bs.clamp(min=0.15); lb = P("log_bw")
        bw = torch.exp(lb[..., :1] + s.m_bwT[:, None, None] * lb[..., 1:] * t)
        kl = s.m_klin[:, None, None]; x0 = P("kx").clamp(max=15.0)                  # option klin: linear beyond x0
        knee = lambda x: (1 - kl) * torch.exp(torch.clamp(x, max=15.0)) + kl * torch.where(
            x < x0, torch.exp(torch.minimum(x, x0)), torch.exp(x0) * (1 + x - x0))
        hsat = 10.0 * (knee((u - Bs) / bw) - knee((-u - Bs) / bw)) * (1 - s.m_noknee[:, None, None])
        e2 = lambda x: torch.exp(torch.clamp(x, max=15.0))
        for kn, on in (("k2", torch.maximum(s.m_knee2, s.m_knee3)), ("k3", s.m_knee3)):   # options knee2, knee3: further knees
            if on.any():
                kp = P(kn); kd = s.m_kpos[:, None, None]
                Bs2 = Bs + kd * nn.functional.softplus(20 * kp[..., :1]) / 20 + (1 - kd) * kp[..., :1]
                bw2 = torch.exp(kp[..., 1:2]).clamp(min=5e-3)
                hsat = hsat + on[:, None, None] * 10.0 * (e2((u - Bs2) / bw2) - e2((-u - Bs2) / bw2))
        tpc = lambda o: torch.cat([tp[..., :1].expand(E, -1, -1), tp[..., 1:] * (1 - o[:, None, None])], -1)   # options a3c, dwc
        a3 = 100 * s.m_cubic[:, None, None] * (P("a3") * tpc(s.m_a3c)).sum(-1, keepdim=True)
        kex = P("kex"); kex = 100 * s.m_exc[:, None, None] * (kex[..., :1] + kex[..., 1:] * t)   # option exc (Bertotti)
        static = a1 * u + a3 * u ** 3 + hsat + kex * (d1 * (d1 ** 2 + 1e-10) ** -0.25)[None]
        lr0 = P("log_r"); rg = s.m_rgeo[:, None, None]                          # option rgeo: log-equidistant widths
        lr0 = rg * (lr0[..., :1] + (lr0[..., -1:] - lr0[..., :1]) * torch.linspace(0, 1, M, device=lr0.device)) + (1 - rg) * lr0
        r = torch.exp(lr0 + (P("rT") * tp[..., 1:]).sum(-1, keepdim=True))                 # (E,N,M)
        alpha = torch.sigmoid(P("alpha") + (P("aT") * tp[..., 1:]).sum(-1, keepdim=True))   # (E,N,1)
        ri = s.m_rateind[:, None, None]; alpha = ri + (1 - ri) * alpha          # rateind: classical play
        tpw = torch.cat([tp[..., :2].expand(E, -1, -1), tp[..., 2:] * (1 - s.m_wlin[:, None, None])], -1)   # option wlin
        w = (P("w") * tpw[..., None, :]).sum(-1)                                 # (E,N,M)
        lr = torch.log(r); x = (lr - lr.mean(-1, keepdim=True)) / (lr.std(-1, keepdim=True) + 1e-6)
        dw = (P("dw") * torch.stack([tp, tpc(s.m_dwc)], -2)).sum(-1)               # (E,N,2); dwc: slope independent of T
        ws = s.m_wsmooth[:, None, None]
        w = ws * (dw[..., :1] + dw[..., 1:] * x) + (1 - ws) * w                 # option wsmooth: density linear in log r
        pas = s.m_passive[:, None, None]                                         # option passive: passive circuit elements
        w = pas * -nn.functional.softplus(-w) + (1 - pas) * w                   # plays: w_i <= 0
        beta = torch.sigmoid(P("beta")) * (1 - 0.5 * pas)                        # (E,N,J); passive: beta <= 0.5
        if dt is not None:
            alpha, beta = rate(alpha), rate(beta)
        cz = (P("cz") * tpw[..., None, :]).sum(-1)                               # (E,N,J)
        cz = pas * -nn.functional.softplus(cz) + (1 - pas) * cz                 # passive: c_j <= 0 (R-L branches)
        cz = cz * (1 - s.m_norel[:, None, None])                                # norel: no relaxations
        gop = P("gop"); gsm = s.m_gsmooth[:, None, None]                        # option gsmooth: play offsets linear in x
        gop = gsm * torch.cat([gop[..., :1] * x, gop[..., M:]], -1) + (1 - gsm) * gop
        gam = P("gam"); gamma = 1e-3 * (1 - s.m_nofb[:, None, None]) * ((gam[..., :1] + gam[..., 1:] * t * (1 - s.m_gamc[:, None, None])) + s.m_fbop[:, None, None] * gop)  # (E,N,M+J)
        u0 = u[..., :1]
        p = u0 - r * torch.tanh(P("g2") * u0 / r + torch.sign(u[..., 5:6] - u0) * P("g"))
        z = u0.expand(E, -1, J)
        if dx0 is not None:
            p = p + dx0[..., :M]; z = z + dx0[..., M:]
        kap = P("kap"); kap = s.m_wmod[:, None, None] * (kap[..., :1] + kap[..., 1:] * t)          # (E,N,1)
        mmod = torch.exp(kap * u ** 2)                                                            # (E,N,W)
        if s.m_rvobs.any() and k >= 50:
            # option rvobs: initial states from the best of G candidate reversal histories (non-causal, whole warm-up),
            # then the trained observer as usual; rvobs = 2 also takes the amplitude factors from that candidate's fit
            p0c, qc = s.cand_init(u, static, hsat, mmod.expand_as(u), r, torch.cat([alpha.expand(-1, -1, M), beta], -1),
                                  torch.cat([r, torch.zeros_like(beta)], -1), torch.cat([w, cz], -1), gamma, HE, k, P)
            rv = ((s.m_rvobs == 1) | (s.m_rvobs == 2)).float()[:, None, None]    # rvobs = 3: prior initial states
            p = rv * p0c[..., :M] + (1 - rv) * p; z = rv * p0c[..., M:] + (1 - rv) * z
        e0 = HE[..., :1] - static[..., :1] - (w * p).sum(-1, keepdim=True) - (cz * z).sum(-1, keepdim=True)
        k0 = s.m_k0[:, None, None]; K0 = P("K0")
        p = p + k0 * 1e-3 * K0[..., :M] * e0; z = z + k0 * 1e-3 * K0[..., M:M + J] * e0
        c = (1 - k0) * e0 + k0 * (P("cpr") + K0[..., -1:] * e0)
        L = gain * P("L")
        cat = lambda a, b: torch.cat([a.expand(*b.shape[:2], -1), b], -1)
        if getattr(s, "capture", False):                                       # law terms for external estimators (revinit.py)
            s.cap = dict(a1=a1, a3=a3, Bs=Bs, bw=bw, u=u, static=static, hsat=hsat, r=r, alpha=alpha, beta=beta, w=w, cz=cz, gamma=gamma, mmod=mmod.expand_as(u), hla=hla if s.m_hlag.any() else torch.zeros_like(a1))
        run = lambda W, st, mm, kk: scan(u[..., :W], st[..., :W], HE[..., :W], torch.cat([p, z], -1), c[..., 0], cat(alpha.expand(-1, -1, M), beta),
                    torch.cat([r, torch.zeros_like(beta)], -1), torch.cat([w, cz], -1), 1e-3 * L[..., :M + J], L[..., -1], kk, gamma,
                    (s.m_fbsat[:, None, None] * hsat)[..., :W], s.m_gate[:, None].expand(-1, len(mid)), m=mm[..., :W])
        if s.m_revinit.any():
            return s.revinit(u, static, hsat, mmod.expand_as(u), r, cat(alpha.expand(-1, -1, M), beta),
                             torch.cat([r, torch.zeros_like(beta)], -1), torch.cat([w, cz], -1), gamma, H, k, P)
        if s.m_qest.any() and k >= getattr(s, "qkmin", 50):
            # option qest: hidden amplitude factor of the window. Open-loop pass over the warm-up, least-squares gain of
            # the (offset- and trend-free) mismatch on the prediction, shrunk by lq; the main pass scales H - c by 1 + q.
            # qsplit > 0: the observer runs over the first qsplit*k samples and q is fitted on the rest only
            # qfit > 0: open-loop pass as usual, but q is fitted on the last (1 - qfit) k samples only
            k0q = max(1, int(getattr(s, "qsplit", 0.0) * k)); k1 = max(k0q, int(getattr(s, "qfit", 0.0) * k))
            Ho = run(k, static, mmod, k0q)
            tt = torch.linspace(-1, 1, k - k1, device=u.device)
            dtr = lambda x: (lambda y: y - (y * tt).sum(-1, keepdim=True) / (tt * tt).sum() * tt)(x - x.mean(-1, keepdim=True))
            hd = dtr(Ho[..., k1:] - c); yd = dtr(HE[..., k1:k] - Ho[..., k1:])
            # Bayesian ridge: prior q ~ N(0, s_q^2) worth n0 = exp(lq) samples of the window's mean-square warm-up H
            n0 = torch.exp(P("lq") + s.m_lqk[:, None, None] * P("lqk") * float(np.log(k / 300))); Sh = (dtr(HE[..., k1:k]) ** 2).mean(-1, keepdim=True)
            q = (yd * hd).sum(-1, keepdim=True) / ((hd * hd).sum(-1, keepdim=True) + n0 * Sh + 1e-6)
            q = s.m_qest[:, None, None] * q.clamp(-0.5, 0.5); s.last_q = q
            if getattr(s, "q_force", None) is not None: q = s.q_force                  # diagnostics: imposed q (E or 1, N, 1)
            qh = q
            if s.m_qest2.any():
                # option qest2: separate gains of the reversible curve (q) and of the operators (qh), 2x2 Bayesian ridge
                h1 = dtr(static[..., k1:k] - hsat[..., k1:k]); h2 = dtr(Ho[..., k1:] - static[..., k1:k] - c)
                s11, s22, s12 = ((a * b).sum(-1, keepdim=True) for a, b in ((h1, h1), (h2, h2), (h1, h2)))
                s11, s22 = s11 + n0 * Sh + 1e-6, s22 + n0 * Sh + 1e-6
                b1, b2 = (yd * h1).sum(-1, keepdim=True), (yd * h2).sum(-1, keepdim=True); det = s11 * s22 - s12 ** 2
                q2 = s.m_qest2[:, None, None]
                q = q2 * ((s22 * b1 - s12 * b2) / det).clamp(-0.5, 0.5) + (1 - q2) * q
                qh = q2 * ((s11 * b2 - s12 * b1) / det).clamp(-0.5, 0.5) + (1 - q2) * qh
                s.last_q = torch.cat([q, qh], -1)
            if s.m_rvobs.any():
                r2 = (s.m_rvobs >= 2).float()[:, None, None]
                q = r2 * qc[..., :1] + (1 - r2) * q; qh = r2 * qc[..., 1:] + (1 - r2) * qh; s.last_q = torch.cat([q, qh], -1)
            static, mmod = static + q * (static - hsat), mmod * (1 + qh)           # saturation knee not scaled
        out = run(u.shape[-1], static, mmod.expand_as(u), k)
        if s.m_hlag.any():
            h1 = ((s.m_hlag == 1) | (s.m_hlag == 3)).float()[:, None, None]; h2 = (s.m_hlag == 2).float()[:, None, None]
            o2 = torch.cat([out[..., :k], s.lag(out[..., k:], hla, H[None, :, k - 1:k])], -1)   # filter the prediction only
            out = h1 * s.lag(out, hla, H[None, :, :1]) + h2 * o2 + (1 - h1 - h2) * out
        return out

    @staticmethod
    def lag(x, a, y0):
        """First-order low-pass y_n = a y_(n-1) + (1 - a) x_n with y_(-1) = y0, along the last axis (FFT convolution)."""
        W = x.shape[-1]; n = torch.arange(W, device=x.device, dtype=x.dtype)
        pw = torch.exp(n * torch.log(a + 1e-12))                                  # a^n
        y = torch.fft.irfft(torch.fft.rfft(x, n=2 * W) * torch.fft.rfft((1 - a) * pw, n=2 * W), n=2 * W)[..., :W]
        return y + a * pw * y0


    def cand_init(s, u, static, hsat, mm, r, alpha, rr, ww, gamma, HE, k, P):
        """Option rvobs: candidate histories as in revinit (drive last reversed at v0 - sgn D, D on a grid 0..2 r_max),
        each run open-loop over the warm-up with the offset c and amplitude factors q, qh fitted in closed form; returns
        the initial states (E,N,M+J) of the best candidate and its (q, qh) (E,N,2), refitted with gradients."""
        E, N, W = u.shape; M = r.shape[-1]; G = getattr(s, "G", 12)
        Hk = HE[..., :k].expand(E, -1, -1)
        v0 = u[..., :1] + gamma * Hk[..., :1]                                     # (E,N,M+J)
        sgn = torch.sign(u[..., 5:6] - u[..., :1]); sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        D = torch.linspace(0, 1, G, device=u.device)[:, None, None, None] * 2 * r.max(-1, keepdim=True)[0][None]
        vp = v0[None, ..., :M]; Bs = vp - sgn[None] * D
        pp = torch.where(sgn[None] > 0, torch.maximum(Bs + r[None], vp - r[None]), torch.minimum(Bs - r[None], vp + r[None]))
        p0 = torch.cat([pp, v0[None, ..., M:].expand(G, -1, -1, -1)], -1)        # (G,E,N,M+J)
        zero = torch.zeros(E, N, device=u.device)
        run = lambda p0_: scan(u[..., :k], static[..., :k], Hk, p0_, zero, alpha, rr, ww, torch.zeros_like(ww), zero, 1, gamma,
                               None, None, mm[..., :k])
        n0 = torch.exp(P("lq") + s.m_lqk[:, None, None] * P("lqk") * float(np.log(k / 300)))
        Sh = (Hk - Hk.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True)

        def fit(ho):
            y = Hk - ho; h1 = (static - hsat)[..., :k].expand_as(ho); h2 = ho - static[..., :k]
            yc, a1, a2 = (x - x.mean(-1, keepdim=True) for x in (y, h1, h2))
            s11 = (a1 * a1).sum(-1, keepdim=True) + n0 * Sh + 1e-6; s22 = (a2 * a2).sum(-1, keepdim=True) + n0 * Sh + 1e-6
            s12 = (a1 * a2).sum(-1, keepdim=True); b1 = (yc * a1).sum(-1, keepdim=True); b2 = (yc * a2).sum(-1, keepdim=True)
            det = s11 * s22 - s12 ** 2
            q = ((s22 * b1 - s12 * b2) / det).clamp(-0.5, 0.5); qh = ((s11 * b2 - s12 * b1) / det).clamp(-0.5, 0.5)
            c = (y - q * h1 - qh * h2).mean(-1, keepdim=True)
            return torch.cat([q, qh], -1), ((y - c - q * h1 - qh * h2) ** 2).mean(-1)

        with torch.no_grad():
            g = torch.stack([fit(run(p0[i]))[1] for i in range(G)]).argmin(0)                # (E,N)
        p0 = torch.gather(p0, 0, g[None, ..., None].expand(1, -1, -1, p0.shape[-1]))[0]
        s.last_g = g
        return p0, fit(run(p0))[0]

    def revinit(s, u, static, hsat, mm, r, alpha, rr, ww, gamma, H, k, P):
        """Option revinit: non-causal warm-up estimator without observer gains; the offset c is not fed back (feedback of
        H - c). Candidate histories: the drive last reversed at B* = v0 - sgn D (D on a grid 0..2 r_max) after coming from
        far away, which fixes every play state; relaxations start at v0. Each candidate runs open-loop over the warm-up
        and the offset c and amplitude factors q (reversible curve), qh (operators) are fitted in closed form (ridge prior
        on q, qh); the candidate with the smallest warm-up error is chosen without gradients, rerun differentiably, refitted
        and continued over the window."""
        E, N, W = u.shape; M = r.shape[-1]; G = getattr(s, "G", 12)
        Hm = H[None, :, :k].mean(-1, keepdim=True)                                # (1,N,1) offset guess for the drive
        v0 = u[..., :1] + gamma * (H[None, :, :1] - Hm)                           # (E,N,M+J)
        sgn = torch.sign(u[..., 5:6] - u[..., :1]); sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        D = torch.linspace(0, 1, G, device=u.device)[:, None, None, None] * 2 * r.max(-1, keepdim=True)[0][None]
        vp = v0[None, ..., :M]; Bs = vp - sgn[None] * D
        pp = torch.where(sgn[None] > 0, torch.maximum(Bs + r[None], vp - r[None]), torch.minimum(Bs - r[None], vp + r[None]))
        p0 = torch.cat([pp, v0[None, ..., M:].expand(G, -1, -1, -1)], -1)        # (G,E,N,M+J)
        Hc = (H[None] - Hm).expand(E, -1, -1)
        zero = torch.zeros(E, N, device=u.device)
        run = lambda W_, st, mmod, p0_: scan(u[..., :W_], st[..., :W_], Hc[..., :W_], p0_, zero, alpha, rr, ww,
                                             torch.zeros_like(ww), zero, 1, gamma, None, None, mmod[..., :W_])
        n0 = torch.exp(P("lq") + s.m_lqk[:, None, None] * P("lqk") * float(np.log(k / 300)))   # (E,N,1)
        Sh = (H[None, :, :k] - Hm).pow(2).mean(-1, keepdim=True)

        def fit(ho, st):
            """Closed-form c, q, qh and warm-up error for open-loop outputs ho (..., k) (offset-free)."""
            y = H[None, :, :k] - ho; h1 = (static - hsat)[..., :k].expand_as(ho); h2 = ho - st[..., :k]
            yc, a1, a2 = (x - x.mean(-1, keepdim=True) for x in (y, h1, h2))
            s11 = (a1 * a1).sum(-1, keepdim=True) + n0 * Sh + 1e-6; s22 = (a2 * a2).sum(-1, keepdim=True) + n0 * Sh + 1e-6
            s12 = (a1 * a2).sum(-1, keepdim=True); b1 = (yc * a1).sum(-1, keepdim=True); b2 = (yc * a2).sum(-1, keepdim=True)
            det = s11 * s22 - s12 ** 2
            q = ((s22 * b1 - s12 * b2) / det).clamp(-0.5, 0.5); qh = ((s11 * b2 - s12 * b1) / det).clamp(-0.5, 0.5)
            c = (y - q * h1 - qh * h2).mean(-1, keepdim=True)
            return c, q, qh, ((y - c - q * h1 - qh * h2) ** 2).mean(-1)

        with torch.no_grad():                                                    # candidates one at a time (memory)
            err = torch.stack([fit(run(k, static, mm, p0[i]), static)[3] for i in range(G)])   # (G,E,N)
            g = err.argmin(0)
        p0 = torch.gather(p0, 0, g[None, ..., None].expand(1, -1, -1, p0.shape[-1]))[0]
        c, q, qh, _ = fit(run(k, static, mm, p0), static)
        s.last_q = torch.cat([q, qh], -1); s.last_g = g
        return run(W, static + q * (static - hsat), mm * (1 + qh), p0) + c


def evaluate_M(m, w, mid, chunk=1024):
    res = {}
    with torch.no_grad():
        w_all = w
        for sc, k in SCENARIOS.items():
            S, Nn = [], []; w = select_k(w_all, k)
            for i0 in range(0, len(w["B"]), chunk):
                B = torch.tensor(w["B"][i0:i0 + chunk], device=DEV); H = torch.tensor(w["H"][i0:i0 + chunk], device=DEV)
                T = torch.tensor(w["T"][i0:i0 + chunk], device=DEV); mi = torch.full((len(B),), mid, device=DEV)
                dt = torch.tensor(w["dt"][i0:i0 + chunk], device=DEV) if "dt" in w else None
                Hp = m(B, H, T, mi, k, dt=dt).cpu().numpy(); Hn = H.cpu().numpy(); Bn = B.cpu().numpy()
                sr, ne = zip(*[metrics(Hp[e], Hn, Bn, k) for e in range(m.E)])
                S.append(np.stack(sr)); Nn.append(np.stack(ne))
            res[sc] = (np.concatenate(S, 1), np.concatenate(Nn, 1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="JSON list of {name, shared: [...], fbop}")
    ap.add_argument("--mats", default="3C94,3C90")
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--J", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=2048, help="windows per material per step")
    ap.add_argument("--win", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--ks", default="100,200,300,450")
    ap.add_argument("--name", default="cm")
    ap.add_argument("--seed", type=int, default=0, help="> 0: other sampling order and 2 %% init jitter of log_r, w")
    ap.add_argument("--qreg", type=float, default=0.0, help="option qest: penalty weight of q^2 (law carries the mean amplitude)")
    ap.add_argument("--accum", type=int, default=1, help="split each step's batch into this many chunks (same gradient, less memory)")
    ap.add_argument("--energy_w", type=float, default=0.0, help="weight of the NERE term in the loss (default of the per-variant option energy_w)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    variants = json.load(open(args.variants)); names = [v["name"] for v in variants]; mats = args.mats.split(",")
    m = CompactM(variants, len(mats), args.M, args.J).to(DEV)
    parts_all = [split_mat(mt)["train"] for mt in mats]
    with torch.no_grad():  # material scale for the initial values: least-squares slope H ~ a B on training data
        for i, pa in enumerate(parts_all):
            Bt = np.concatenate([d["B"][:50, ::7].ravel() for d in pa]); Ht = np.concatenate([d["H"][:50, ::7].ravel() for d in pa])
            sc = float((Bt * Ht).sum() / (Bt * Bt).sum()) / 180.0
            m.p["a1"][:, i, 0] *= sc; m.p["w"][:, i, :, 0] *= sc; m.p["dw"][:, i, 0, 0] *= sc
            print(f"{mats[i]}: initial scale {sc:.2f}", flush=True)
        if args.seed > 0:
            m.p["log_r"] += 0.02 * torch.randn_like(m.p["log_r"]); m.p["w"][..., 0] *= 1 + 0.02 * torch.randn_like(m.p["w"][..., 0])
    npar = m.n_params()
    print(f"device {DEV}; materials {mats}; variants: " + ", ".join(f"{n}({p})" for n, p in zip(names, npar)), flush=True)
    samplers = [GPUSampler(pa, DEV) for pa in parts_all]
    vws = [{k: v[np.random.default_rng(0).permutation(len(v))][:350] for k, v in windows("val", mt).items()} for mt in mats]
    opt = torch.optim.Adam(m.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    ew = torch.tensor([float(v.get("energy_w", args.energy_w)) for v in variants], device=DEV)[:, None]   # per variant
    gen = torch.Generator(DEV).manual_seed(1 + 1000 * args.seed); rng = np.random.default_rng(1 + 1000 * args.seed); t0 = time.time()
    for it in range(args.steps):
        bt = [smp(args.bs, args.win, gen) for smp in samplers]
        B, H, T = (torch.cat(x) for x in zip(*bt))
        mid = torch.cat([torch.full((len(b[0]),), i, device=DEV) for i, b in enumerate(bt)])
        dts = [smp.last_dt for smp in samplers]
        dt = torch.cat([d if d is not None else torch.ones(len(b[0]), device=DEV) for d, b in zip(dts, bt)]) if any(d is not None for d in dts) else None
        k = int(rng.choice([int(x) for x in args.ks.split(",")]))
        pos = torch.cat([torch.arange(len(b[0]), device=DEV) for b in bt])   # index of a window within its material
        opt.zero_grad(); lossE = 0
        for c in range(args.accum):   # each chunk holds 1/accum of every material's windows: the chunk losses average to the full one
            sel = pos % args.accum == c; Bc, Hc, mc = B[sel], H[sel], mid[sel]
            Hp = m(Bc, Hc, T[sel], mc, k, dt=None if dt is None else dt[sel])
            sre = sre_E(Hp, Hc, k)                                                  # (E,N)
            if (ew > 0).any():   # NERE as evaluated: energy error of the predicted part / sum |H dB| over the window
                dB = torch.gradient(Bc, dim=1)[0][None]
                nere = ((Hp[..., k:] - Hc[None, :, k:]) * dB[..., k:]).sum(-1).abs() / (Hc[None] * dB).abs().sum(-1).clamp(min=1e-9)
                sre = sre + ew * nere
            if args.qreg > 0 and getattr(m, "last_q", None) is not None:
                sre = sre + args.qreg * (m.last_q ** 2).mean(-1); m.last_q = None
            lc = torch.stack([sre[:, mc == i].mean(1) for i in range(len(mats))], 1) / args.accum   # (E,nmat)
            lc.sum().backward(); lossE = lossE + lc.detach()
        for p in m.parameters():                     # a diverged (variant, material) must not turn its parameters into NaN
            if p.grad is not None:
                p.grad.nan_to_num_(0.0, 0.0, 0.0)
        opt.step(); sched.step()
        if it % 250 == 0:
            print(f"it {it} loss " + " ".join("/".join(f"{100*x:.2f}" for x in row) for row in lossE.tolist()) + f"  t {time.time()-t0:.0f}s", flush=True)
        if (it % 2000 == 0 and it > 0) or it == args.steps - 1:
            for i, mt in enumerate(mats):
                summarize(evaluate_M(m, vws[i], i), names, npar, f"val {mt}")
            torch.save({"variants": variants, "names": names, "mats": mats, "M": args.M, "J": args.J, "sd": m.state_dict()}, ROOT / f"models/{args.name}.pt")
