"""Gain-scheduled circuit (as the thermal neural network scales a lumped-parameter thermal network): per material, an MLP
maps the excitation at every step to shifts of the element values of the classical circuit, which then pass through the
same constraint transforms as in the classical circuit (passive at every step). The backbone is the classical circuit
(nere:final_e1 option set); its forward is re-implemented here with per-step coefficients.

Inputs per step n (exogenous, computed before the recurrence): u_n / 0.2 T, asinh(dB_n / 1 mT), asinh(d2B_n / 0.1 mT),
t = (T - 50) / 25. MLP: 4 -> h -> h -> 16 (tanh, tanh, linear). The output is made even in (u, dB, d2B),
z(x) = (f(x) + f(-x)) / 2, so that the model stays odd-symmetric, H(-B) = -H(B). Shifts (last layer zero-initialized:
training starts at the classical circuit):
  static part (5):   a1 * e^z, a3 + 100 z, Bs + 0.05 z, b * e^z, hysteresis sum * e^z
  dynamics (11):     logit alpha + z; log r_i + z + z' chi_i; play density d0 + z, d1 + z;
                     logit beta_j + z_j; c_j raw + z_j; gamma + 1e-3 (z + z' chi_i)        (per-step kernel: scan_tv.py)
The warm-up estimator is the classical circuit's. Parameters per material: 32 + MLP (h = 16: 624, h = 64: 5520).
Usage: python hyper_m.py --init nere:final_e1 --hidden 16 --joint --name <out>   -> models/<out>.pt
(--opt AB is accepted for the scripts of the paper: the dynamics and the static part are scheduled together)"""
import argparse, time
import numpy as np, torch, torch.nn as nn
from mc2lib import ROOT, split_mat, windows
from common import GPUSampler, sre_E, summarize
from compact_m import CompactM, evaluate_M, DEV
import scan_tv
scan_k = scan_tv.scan if DEV.type == "cuda" else scan_tv.scan_ref   # no GPU: the plain torch reference (as compact_m)

NIN = 4
OUTS = {"A": ["a1", "a3", "bs", "bw", "m"], "B": ["al", "r0", "r1", "d0", "d1", "be1", "be2", "c1", "c2", "g0", "g1"]}
FINAL = {"fbop": 1, "k0": 1, "cubic": 1, "wsmooth": 1, "qest": 1, "lqk": 1, "qest2": 1, "rgeo": 1, "gsmooth": 1, "tlin": 1,
         "passive": 1, "hlag": 3}


outs_b = lambda J: OUTS["B"][:5] + [f"be{j + 1}" for j in range(J)] + [f"c{j + 1}" for j in range(J)] + OUTS["B"][-2:]   # J relaxations


class HyperM(nn.Module):
    """Gain-scheduled circuit: the classical circuit (bb) with per-material scheduling MLPs on its element values."""
    sg: torch.Tensor                                                            # input signs of the mirrored branch
    def __init__(s, variant, mats, M, J, hidden, opt="AB"):
        """variant: option set of the classical circuit; mats: materials; M, J: plays, relaxations; hidden: MLP width."""
        super().__init__()
        assert {k: v for k, v in variant.items() if k not in ("name", "shared", "energy_w")} == FINAL, "backbone: final option set only"
        assert opt == "AB", "the static part and the dynamics are scheduled together (option AB)"
        s.bb = CompactM([variant], len(mats), M, J); s.E, s.nm, s.hidden, s.M, s.J, s.opt = 1, len(mats), hidden, M, J, opt
        s.outs = OUTS["A"] + outs_b(J)
        nm, h, no = len(mats), hidden, len(s.outs)
        u = lambda *sh, fan: nn.Parameter(torch.empty(*sh).uniform_(-fan ** -0.5, fan ** -0.5))
        s.w1, s.b1 = u(nm, NIN, h, fan=NIN), u(nm, h, fan=NIN)
        s.w2, s.b2 = u(nm, h, h, fan=h), u(nm, h, fan=h)
        s.w3, s.b3 = nn.Parameter(torch.zeros(nm, h, no)), nn.Parameter(torch.zeros(nm, no))
        s.register_buffer("sg", torch.tensor([-1.0, -1.0, -1.0, 1.0]), persistent=False)  # input signs of the mirrored branch
        s.last_q = None; s.zero = False; s.on_dev = False                     # zero: schedule switched off (tests)

    def n_params(s):
        """Parameters of the (single) variant: the circuit's per material plus the MLPs."""
        per = sum(p[0].numel() for p in (s.w1, s.b1, s.w2, s.b2, s.w3, s.b3))
        return [s.bb.n_params()[0] + s.nm * per]


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

    def forward(s, B, H, T, mid, k):  # ty: ignore[invalid-mutable-override]  (torch declares forward as an attribute)
        """Classical circuit with per-step element values. B, H (N,W), T (N,). Returns (1,N,W)."""
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
        Z = torch.zeros(N, W, len(s.outs), device=B.device) if s.zero else s.hyper(u[0], d1, d2, (T - 50) / 25, mid)
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
        zs = lambda o: z[o][..., None]                                          # per-step operator coefficients (1,N,W,K)
        alpha = torch.sigmoid(al0[:, :, None] + zs("al"))
        r = torch.exp(lr[:, :, None] + zs("r0") + zs("r1") * x[:, :, None])
        w = -nn.functional.softplus(-((dw[..., None, :1] + zs("d0")) + (dw[..., None, 1:] + zs("d1")) * x[:, :, None]))
        beta = 0.5 * torch.sigmoid(P("beta")[:, :, None] + torch.cat([zs(f"be{j + 1}") for j in range(J)], -1))
        cz = -nn.functional.softplus(cz0[:, :, None] + torch.cat([zs(f"c{j + 1}") for j in range(J)], -1))
        xg = torch.cat([x, torch.zeros_like(x[..., :J])], -1)[:, :, None]
        gamma = gamma[:, :, None] + 1e-3 * (zs("g0") + zs("g1") * xg)
        r0, w0, cz0_ = r[:, :, 0], w[:, :, 0], cz[:, :, 0]
        u0 = u[..., :1]
        p = u0 - r0 * torch.tanh(P("g2") * u0 / r0 + torch.sign(u[..., 5:6] - u0) * P("g"))
        zz = u0.expand(1, -1, J)
        e0 = HE[..., :1] - static[..., :1] - (w0 * p).sum(-1, keepdim=True) - (cz0_ * zz).sum(-1, keepdim=True)
        K0 = P("K0")
        p = p + 1e-3 * K0[..., :M] * e0; zz = zz + 1e-3 * K0[..., M:M + J] * e0
        c = P("cpr") + K0[..., -1:] * e0
        L = P("L")
        al_k = torch.cat([alpha.expand(-1, -1, -1, M), beta], -1); r_k = torch.cat([r, torch.zeros_like(beta)], -1)
        w_k = torch.cat([w, cz], -1); sl = lambda v, Wn: v[..., :Wn, :]
        run = lambda Wn, st, mm, kk: scan_k(u[..., :Wn], st[..., :Wn], HE[..., :Wn], torch.cat([p, zz], -1), c[..., 0],
                                                 sl(al_k, Wn), sl(r_k, Wn), sl(w_k, Wn), 1e-3 * L[..., :M + J], L[..., -1], kk,
                                                 sl(gamma, Wn), m=mm[..., :Wn])
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
            q = ((s22 * b1 - s12 * b2) / det).clamp(-0.5, 0.5); qh = ((s11 * b2 - s12 * b1) / det).clamp(-0.5, 0.5)
            s.last_q = torch.cat([q, qh], -1)
            static, mmod = static + q * (static - hsat), mmod * (1 + qh)
        out = run(W, static, mmod, k)
        return CompactM.lag(out, hla, H[None, :, :1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="<ckpt>:<variant> of the backbone (final option set)")
    ap.add_argument("--opt", default="AB", choices=["AB"], help="scheduled: static part and dynamics (the only option)")
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
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    ckn, var = args.init.split(":")
    src = torch.load(ROOT / f"models/{ckn}.pt", map_location=DEV); e0 = src["names"].index(var); mats = src["mats"]
    variant = src["variants"][e0]
    m = HyperM(variant, mats, src["M"], src["J"], args.hidden).to(DEV)
    m.bb.load_state_dict({k: v[e0:e0 + 1] for k, v in src["sd"].items() if k.startswith("p.")}, strict=False)
    if not args.joint:
        for p in m.bb.parameters():
            p.requires_grad_(False)
    samplers = [GPUSampler(split_mat(mt)["train"], DEV) for mt in mats]
    gen = torch.Generator(DEV).manual_seed(1 + 1000 * args.seed); rng = np.random.default_rng(1 + 1000 * args.seed)
    npar = m.n_params(); names = [f"{var}+hyperAB{args.hidden}" + ("j" if args.joint else "")]
    print(f"device {DEV}; materials {mats}; {names[0]}: {npar[0] // len(mats)} params per material; outputs {m.outs}", flush=True)
    vws = [{k: v[np.random.default_rng(0).permutation(len(v))][:350] for k, v in windows("val", mt).items()} for mt in mats]
    for i, mt in enumerate(mats):
        summarize(evaluate_M(m, vws[i], i), names, [n // len(mats) for n in npar], f"val {mt} (backbone)")
    groups = [{"params": [p for n, p in m.named_parameters() if not n.startswith("bb.")], "lr": args.lr}]
    if args.joint:
        groups.append({"params": list(m.bb.parameters()), "lr": args.lr_bb})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr] + ([args.lr_bb] if args.joint else []), total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for it in range(args.steps):
        bt = [smp(args.bs, args.win, gen) for smp in samplers]
        B, H, T = (torch.cat(x) for x in zip(*bt))
        mid = torch.cat([torch.full((len(b[0]),), i, device=DEV) for i, b in enumerate(bt)])
        pos = torch.cat([torch.arange(len(b[0]), device=DEV) for b in bt])
        k = int(rng.choice([int(x) for x in args.ks.split(",")]))
        opt.zero_grad(); lossE = torch.zeros((), device=DEV)
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
                        "hidden": args.hidden, "opt": "AB", "sd": m.state_dict(), "init": args.init}, ROOT / f"models/{args.name}.pt")
