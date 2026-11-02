"""The classical circuit as a multi-material PyTorch model (training: python compact_m.py --variants <json> ...).

Law (one sample = 62.5 ns, t = (T - 50)/25, coefficients polynomial in t):
    u   = B + tau1(T) B' + tau2(T) B''                             measurement correction (16 MHz acquisition chain)
    v_i = u + gamma_i(T) H_prev                                     drive of operator i (mean-field feedback of H)
    z_j <- z_j + beta_j (v_j - z_j)                                 linear relaxations
    p_i <- p_i + alpha(T) dz_{r_i(T)}(v_i - p_i)                    viscous play operators
    H   = a1(T) u + a3(T) u^3 + 10 A/m [e^{(u-Bs(T))/b} - e^{(-u-Bs(T))/b}] + sum w_i(T) p_i + sum c_j(T) z_j + c
    H_meas = first-order lag of H (pole <= 0.6 per material, option hlag)
Variants (leading axis E) differ in which parameter names are shared across materials ("shared" list) and in the
options (all used by configs/*.json):
  cubic (a3 u^3 term), nofb (no H feedback), fbop (per-operator feedback offsets), gsmooth (play feedback offsets linear
  in the standardized log r: gop_i = gop_0 x_i), rgeo (play widths log-equidistant between the first and last log_r),
  wsmooth (play weights w_i = d0(T) + d1(T) x_i), tlin (all coefficients at most linear in T), passive (positive
  circuit elements apart from the feedback: w_i <= 0, c_j <= 0, beta_j <= 0.5), hlag (= 3: first-order lag of the H
  measurement, pole <= 0.6), tauc, gamc, a3c, dwc (measurement correction, feedback gain, cubic coefficient, slope of
  the play density independent of T), and the ablations noknee (no saturation knee), rateind (rate-independent plays,
  alpha = 1), norel (no relaxations), notau (no measurement correction, u = B).
Warm-up estimator (per material, not counted): prior initial play states (g, g2), initial mismatch spread over all
states (option k0: gains K0, offset prior cpr), observer gains L, and the amplitude factors of the window (qest: one
factor, qest2: separate factors of the reversible curve and of the operators; prior strength lq, lqk: k-dependent).
"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn
from mc2lib import ROOT, SCENARIOS, metrics, select_k, split_mat, windows
from common import GPUSampler, sre_E, summarize
from scan import scan

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu":    # no GPU: the plain torch reference of the fused kernel (same results, much slower)
    from scan import scan_ref as scan
if DEV.type == "cuda":   # the GPU also drives the desktop: leave the compositor ~2.5 GB of VRAM (a full GPU hung the login screen)
    torch.cuda.set_per_process_memory_fraction(0.84)
LAW = ["tau", "a1", "a3", "bs", "log_bw", "log_r", "rT", "alpha", "aT", "w", "beta", "cz", "gam", "gop", "dw", "hl"]


def init_params(M, J):
    """Base shape and initial value of every parameter (without the E and material axes)."""
    return {
        "tau": torch.tensor([[1.6, 0.0], [5.6, 0.0]]),          # [tau1, tau2] x [1, t]
        "a1": torch.tensor([180.0, 0.0, 0.0]),                    # [1, t, t^2]
        "a3": torch.zeros(3),                                     # cubic, x100 A/m/T^3 (option "cubic")
        "bs": torch.tensor([0.40, 0.0, 0.0]), "log_bw": torch.tensor([np.log(0.03), 0.0]),   # log_bw: only [0] used
        "log_r": torch.linspace(np.log(0.01), np.log(0.25), M), "rT": torch.zeros(2),
        "alpha": torch.tensor([-1.5]), "aT": torch.zeros(2),
        "w": torch.cat([torch.full((M, 1), -10.0), torch.zeros(M, 2)], 1),
        "beta": torch.tensor([-4.2, -1.4, -0.5][:J]) if J <= 3 else torch.linspace(-5.5, -0.5, J), "cz": torch.zeros(J, 3),
        "gam": torch.zeros(2), "gop": torch.zeros(M + J),
        "dw": torch.tensor([[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # option wsmooth: w_i = dw0(T) + dw1(T) x_i, x = std. log r_i
        "hl": torch.tensor([-3.0]),                               # option hlag: H-measurement low-pass pole a = 0.6 sigmoid(hl)
        "g": torch.zeros(M), "g2": torch.zeros(M), "L": torch.zeros(M + J + 1),
        "K0": torch.cat([torch.zeros(M + J), torch.ones(1)]), "cpr": torch.zeros(1),
        "lq": torch.tensor([float(np.log(300.0))]),               # option qest: log prior strength (samples) of the amplitude estimate
        "lqk": torch.zeros(1),                                    # option lqk: n0 = exp(lq) (k/300)^lqk
    }


OPTIONS = ("fbop", "k0", "nofb", "cubic", "wsmooth", "qest", "tlin", "rgeo", "gsmooth", "lqk", "qest2", "tauc", "passive",
           "gamc", "a3c", "dwc", "hlag", "noknee", "rateind", "norel", "notau")


class CompactM(nn.Module):
    """Classical circuit for E variants and n_mat materials at once (parameters (E, n_mat, ...))."""
    p: nn.ParameterDict                                                         # parameters by name, (E, n_mat, ...)
    # option switches per variant (E,), registered as buffers in __init__ (see OPTIONS)
    m_fbop: torch.Tensor; m_k0: torch.Tensor; m_nofb: torch.Tensor; m_cubic: torch.Tensor; m_wsmooth: torch.Tensor
    m_qest: torch.Tensor; m_tlin: torch.Tensor; m_rgeo: torch.Tensor; m_gsmooth: torch.Tensor; m_lqk: torch.Tensor
    m_qest2: torch.Tensor; m_tauc: torch.Tensor; m_passive: torch.Tensor; m_gamc: torch.Tensor; m_a3c: torch.Tensor
    m_dwc: torch.Tensor; m_hlag: torch.Tensor; m_noknee: torch.Tensor; m_rateind: torch.Tensor; m_norel: torch.Tensor
    m_notau: torch.Tensor
    capture: bool = False; cap: dict                                            # circuit terms for ekf_eval.py
    last_q: torch.Tensor | None = None                                         # amplitude factors of the last call
    def __init__(s, variants, n_mat, M=6, J=2):
        """variants: list of dicts {"name", "shared": [names], <option>: 0/1 (hlag: 0/3), "energy_w", "hlinit"}."""
        super().__init__()
        E = len(variants); s.E, s.M, s.J, s.nm = E, M, J, n_mat
        s.shared = {k: torch.tensor([float(k in v.get("shared", [])) for v in variants]) for k in LAW}
        assert all(v.get("hlag", 0) in (0, 3) for v in variants), "option hlag: 0 or 3"
        for o in OPTIONS:
            s.register_buffer(f"m_{o}", torch.tensor([float(v.get(o, 0)) for v in variants]))
        s.p = nn.ParameterDict({k: nn.Parameter(v.float().expand(E, n_mat, *v.shape).clone()) for k, v in init_params(M, J).items()})
        with torch.no_grad():                                                   # per-variant initial H-lag pole (hlinit)
            for e, v in enumerate(variants):
                if "hlinit" in v: s.p["hl"][e] = float(v["hlinit"])

    def n_params(s):
        """Fitted parameters of each variant: per-material ones once per material, shared ones once."""
        out = []
        for e in range(s.E):
            n = 0
            for k in LAW:
                if k == "gop" and not s.m_fbop[e]: continue
                if k in ("gam", "gop") and s.m_nofb[e]: continue
                if k == "a3" and not s.m_cubic[e]: continue
                if k == "dw" and not s.m_wsmooth[e]: continue
                if k == "hl" and not s.m_hlag[e]: continue
                if (k in ("bs", "log_bw") and s.m_noknee[e]) or (k in ("alpha", "aT") and s.m_rateind[e]) \
                        or (k in ("beta", "cz") and s.m_norel[e]) or (k == "tau" and s.m_notau[e]): continue
                if k == "w" and s.m_wsmooth[e]: continue
                size = 1 if k == "log_bw" else s.p[k][0, 0].numel()
                if k == "log_r" and s.m_rgeo[e]: size = 2
                if k == "tau" and s.m_tauc[e]: size = 2
                if k == "gop" and s.m_gsmooth[e]: size = 1 + s.J * (1 - int(s.m_norel[e]))
                if s.m_tlin[e]:                                                  # no t^2 coefficients
                    if k in ("a1", "a3", "rT", "aT", "bs"): size -= 1
                    if k in ("w", "cz", "dw"): size = size // 3 * 2
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

    def forward(s, B, H, T, mid, k):  # ty: ignore[invalid-mutable-override]  (torch declares forward as an attribute)
        """B, H (N,W) at 16 MHz, T (N,), material index mid (N,); the measured H is used for n < k (warm-up).
        Returns (E,N,W)."""
        E, M, J = s.E, s.M, s.J
        P = lambda n: s.P(n, mid)
        HE = H[None]
        if s.m_hlag.any():
            # option hlag: the H channel of the instrument is a first-order low-pass y_n = a y_(n-1) + (1 - a) H_n of the
            # field H the law computes (a <= 0.6). The warm-up is de-lagged with the exact inverse; the output is filtered.
            hla = (s.m_hlag > 0).float()[:, None, None] * 0.6 * torch.sigmoid(P("hl"))       # (E,N,1)
            HE = (H[None] - hla * torch.cat([H[:, :1], H[:, :-1]], -1)[None]) / (1 - hla)
        t = ((T - 50) / 25)[None, :, None]                                      # (1,N,1)
        tp = torch.cat([torch.ones_like(t), t, t ** 2], -1)                     # (1,N,3)
        tp = torch.cat([tp[..., :2].expand(E, -1, -1), tp[..., 2:] * (1 - s.m_tlin[:, None, None])], -1)   # option tlin: no t^2 terms
        d1 = torch.gradient(B, dim=1)[0]; d2 = torch.gradient(d1, dim=1)[0]
        tpt = torch.cat([tp[..., :1].expand(E, -1, -1), tp[..., 1:2] * (1 - s.m_tauc[:, None, None])], -1)   # option tauc
        tau = (P("tau") * tpt[..., None, :]).sum(-1) * (1 - s.m_notau[:, None, None])   # (E,N,2); notau: u = B
        u = B[None] + tau[..., :1] * d1[None] + tau[..., 1:] * d2[None]       # (E,N,W)
        a1 = (P("a1") * tp).sum(-1, keepdim=True)
        Bs = (P("bs") * tp).sum(-1, keepdim=True).clamp(min=0.15)
        bw = torch.exp(P("log_bw")[..., :1])
        knee = lambda x: torch.exp(torch.clamp(x, max=15.0))
        hsat = 10.0 * (knee((u - Bs) / bw) - knee((-u - Bs) / bw)) * (1 - s.m_noknee[:, None, None])
        tpc = lambda o: torch.cat([tp[..., :1].expand(E, -1, -1), tp[..., 1:] * (1 - o[:, None, None])], -1)   # options a3c, dwc
        a3 = 100 * s.m_cubic[:, None, None] * (P("a3") * tpc(s.m_a3c)).sum(-1, keepdim=True)
        static = a1 * u + a3 * u ** 3 + hsat
        lr0 = P("log_r"); rg = s.m_rgeo[:, None, None]                          # option rgeo: log-equidistant widths
        lr0 = rg * (lr0[..., :1] + (lr0[..., -1:] - lr0[..., :1]) * torch.linspace(0, 1, M, device=lr0.device)) + (1 - rg) * lr0
        r = torch.exp(lr0 + (P("rT") * tp[..., 1:]).sum(-1, keepdim=True))                 # (E,N,M)
        alpha = torch.sigmoid(P("alpha") + (P("aT") * tp[..., 1:]).sum(-1, keepdim=True))   # (E,N,1)
        ri = s.m_rateind[:, None, None]; alpha = ri + (1 - ri) * alpha          # rateind: classical play
        w = (P("w") * tp[..., None, :]).sum(-1)                                  # (E,N,M)
        lr = torch.log(r); x = (lr - lr.mean(-1, keepdim=True)) / (lr.std(-1, keepdim=True) + 1e-6)
        dw = (P("dw") * torch.stack([tp, tpc(s.m_dwc)], -2)).sum(-1)               # (E,N,2); dwc: slope independent of T
        ws = s.m_wsmooth[:, None, None]
        w = ws * (dw[..., :1] + dw[..., 1:] * x) + (1 - ws) * w                 # option wsmooth: density linear in log r
        pas = s.m_passive[:, None, None]                                         # option passive: passive circuit elements
        w = pas * -nn.functional.softplus(-w) + (1 - pas) * w                   # plays: w_i <= 0
        beta = torch.sigmoid(P("beta")) * (1 - 0.5 * pas)                        # (E,N,J); passive: beta <= 0.5
        cz = (P("cz") * tp[..., None, :]).sum(-1)                                # (E,N,J)
        cz = pas * -nn.functional.softplus(cz) + (1 - pas) * cz                 # passive: c_j <= 0 (R-L branches)
        cz = cz * (1 - s.m_norel[:, None, None])                                # norel: no relaxations
        gop = P("gop"); gsm = s.m_gsmooth[:, None, None]                        # option gsmooth: play offsets linear in x
        gop = gsm * torch.cat([gop[..., :1] * x, gop[..., M:]], -1) + (1 - gsm) * gop
        gam = P("gam"); gamma = 1e-3 * (1 - s.m_nofb[:, None, None]) * ((gam[..., :1] + gam[..., 1:] * t * (1 - s.m_gamc[:, None, None])) + s.m_fbop[:, None, None] * gop)  # (E,N,M+J)
        u0 = u[..., :1]
        p = u0 - r * torch.tanh(P("g2") * u0 / r + torch.sign(u[..., 5:6] - u0) * P("g"))
        z = u0.expand(E, -1, J)
        mmod = torch.ones_like(u)                                               # multiplier of the operator sum (1 + qh)
        e0 = HE[..., :1] - static[..., :1] - (w * p).sum(-1, keepdim=True) - (cz * z).sum(-1, keepdim=True)
        k0 = s.m_k0[:, None, None]; K0 = P("K0")
        p = p + k0 * 1e-3 * K0[..., :M] * e0; z = z + k0 * 1e-3 * K0[..., M:M + J] * e0
        c = (1 - k0) * e0 + k0 * (P("cpr") + K0[..., -1:] * e0)
        L = P("L")
        cat = lambda a, b: torch.cat([a.expand(*b.shape[:2], -1), b], -1)
        if s.capture:                                                           # circuit terms for the Kalman filter (ekf_eval.py)
            s.cap = dict(a1=a1, a3=a3, Bs=Bs, bw=bw, u=u, static=static, hsat=hsat, r=r, alpha=alpha, beta=beta, w=w, cz=cz, gamma=gamma, mmod=mmod, hla=hla if s.m_hlag.any() else torch.zeros_like(a1))
        run = lambda W, st, mm, kk: scan(u[..., :W], st[..., :W], HE[..., :W], torch.cat([p, z], -1), c[..., 0], cat(alpha.expand(-1, -1, M), beta),
                    torch.cat([r, torch.zeros_like(beta)], -1), torch.cat([w, cz], -1), 1e-3 * L[..., :M + J], L[..., -1], kk, gamma, m=mm[..., :W])
        if s.m_qest.any() and k >= 50:
            # option qest: hidden amplitude factor of the window. Open-loop pass over the warm-up (observer at the first
            # sample only), least-squares gain of the (offset- and trend-free) mismatch, shrunk by the prior n0 = exp(lq);
            # the main pass scales H - c by 1 + q.
            k1 = 1
            Ho = run(k, static, mmod, 1)
            tt = torch.linspace(-1, 1, k - k1, device=u.device)
            dtr = lambda x: (lambda y: y - (y * tt).sum(-1, keepdim=True) / (tt * tt).sum() * tt)(x - x.mean(-1, keepdim=True))
            hd = dtr(Ho[..., k1:] - c); yd = dtr(HE[..., k1:k] - Ho[..., k1:])
            # Bayesian ridge: prior q ~ N(0, s_q^2) worth n0 = exp(lq) samples of the window's mean-square warm-up H
            n0 = torch.exp(P("lq") + s.m_lqk[:, None, None] * P("lqk") * float(np.log(k / 300))); Sh = (dtr(HE[..., k1:k]) ** 2).mean(-1, keepdim=True)
            q = (yd * hd).sum(-1, keepdim=True) / ((hd * hd).sum(-1, keepdim=True) + n0 * Sh + 1e-6)
            q = s.m_qest[:, None, None] * q.clamp(-0.5, 0.5); s.last_q = q
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
            static, mmod = static + q * (static - hsat), mmod * (1 + qh)           # saturation knee not scaled
        out = run(u.shape[-1], static, mmod.expand_as(u), k)
        if s.m_hlag.any():
            h1 = (s.m_hlag == 3).float()[:, None, None]
            out = h1 * s.lag(out, hla, H[None, :, :1]) + (1 - h1) * out
        return out

    @staticmethod
    def lag(x, a, y0):
        """First-order low-pass y_n = a y_(n-1) + (1 - a) x_n with y_(-1) = y0, along the last axis (FFT convolution)."""
        W = x.shape[-1]; n = torch.arange(W, device=x.device, dtype=x.dtype)
        pw = torch.exp(n * torch.log(a + 1e-12))                                  # a^n
        y = torch.fft.irfft(torch.fft.rfft(x, n=2 * W) * torch.fft.rfft((1 - a) * pw, n=2 * W), n=2 * W)[..., :W]
        return y + a * pw * y0


def evaluate_M(m, w, mid, chunk=1024):
    """Per-window SRE and NERE of every variant on windows w of material index mid: {scenario: (sre, nere)}, each (E, N)."""
    res = {}
    with torch.no_grad():
        w_all = w
        for sc, k in SCENARIOS.items():
            S, Nn = [], []; w = select_k(w_all, k)
            for i0 in range(0, len(w["B"]), chunk):
                B = torch.tensor(w["B"][i0:i0 + chunk], device=DEV); H = torch.tensor(w["H"][i0:i0 + chunk], device=DEV)
                T = torch.tensor(w["T"][i0:i0 + chunk], device=DEV); mi = torch.full((len(B),), mid, device=DEV)
                Hp = m(B, H, T, mi, k).cpu().numpy(); Hn = H.cpu().numpy(); Bn = B.cpu().numpy()
                sr, ne = zip(*[metrics(Hp[e], Hn, Bn, k) for e in range(m.E)])
                S.append(np.stack(sr)); Nn.append(np.stack(ne))
            res[sc] = (np.concatenate(S, 1), np.concatenate(Nn, 1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="JSON list of variants {name, shared: [...], <option>: 0/1}")
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
        k = int(rng.choice([int(x) for x in args.ks.split(",")]))
        pos = torch.cat([torch.arange(len(b[0]), device=DEV) for b in bt])   # index of a window within its material
        opt.zero_grad(); lossE = torch.zeros((), device=DEV)
        for c in range(args.accum):   # each chunk holds 1/accum of every material's windows: the chunk losses average to the full one
            sel = pos % args.accum == c; Bc, Hc, mc = B[sel], H[sel], mid[sel]
            Hp = m(Bc, Hc, T[sel], mc, k)
            sre = sre_E(Hp, Hc, k)                                                  # (E,N)
            if (ew > 0).any():   # NERE as evaluated: energy error of the predicted part / sum |H dB| over the window
                dB = torch.gradient(Bc, dim=1)[0][None]
                nere = ((Hp[..., k:] - Hc[None, :, k:]) * dB[..., k:]).sum(-1).abs() / (Hc[None] * dB).abs().sum(-1).clamp(min=1e-9)
                sre = sre + ew * nere
            q = m.last_q
            if args.qreg > 0 and q is not None:
                sre = sre + args.qreg * (q ** 2).mean(-1); m.last_q = None
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
