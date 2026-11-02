"""Analysis of a gain-scheduled model (hyper_m.py, options with B): per-step coefficients and the energy balance of its
equivalent circuit on the test windows.

Energy check. At every step the scheduled coefficients satisfy the same constraints as the classical model (w, c <= 0,
alpha in (0, 1), beta in (0, 1/2], r > 0), so every element of the circuit is positive at every instant and the sliders and
relaxation resistors dissipate, D_n = sum_i m |w_i| (v_i - (p_i,n + p_i,n-1)/2)(p_i,n - p_i,n-1) >= 0. A time-varying
inductor is not passive, however: changing its value at a fixed state changes its stored energy (parametric pumping).
The pumped energy per step is E(x_(n-1); theta_n) - E(x_(n-1); theta_(n-1)) with the stored energy
E = W_s(u) + sum_i m w_i v_i^2 / 2 + sum_i m |w_i| (v_i - p_i)^2 / 2 (reversible curve, parallel and series inductors of
plays and relaxations; per unit core volume). The feedback sources are active in the classical circuit already and are not
counted. For the classical model the pumped energy is zero. Reported over the predicted part (n >= k, open loop) of the
test windows: pumped energy (net and positive part) relative to the dissipated energy.

Usage: python hyper_analysis.py <ckpt> [--k 500] [--n 300]   -> results/hyper_energy_<ckpt>.json, results/hyper_coef_<ckpt>.npz"""
import argparse, json
import numpy as np, torch
from mc2lib import ROOT, windows
from compact_m import DEV
from loader import load
import scan_tv

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--k", type=int, default=500); ap.add_argument("--n", type=int, default=300)
args = ap.parse_args()
m, names, mats = load(args.ckpt); m.eval()
cap = {}
_scan, _hyper = scan_tv.scan, m.hyper
scan_tv.scan = lambda *a, **kw: (cap.__setitem__("scan", (a, kw)), _scan(*a, **kw))[1]  # ty: ignore[invalid-assignment]  (records the kernel inputs)
m.hyper = lambda *a: (lambda z: (cap.__setitem__("z", z), z)[1])(_hyper(*a))


def W_rev(u, a1, a3, Bs, bw):
    """Energy of the reversible curve h_s, integral of h_s from 0 to u (H_kn = 10 A/m)."""
    e = lambda x: torch.exp(torch.clamp(x, max=15.0))
    return a1 * u ** 2 / 2 + a3 * u ** 4 / 4 + 10 * bw * (e((u - Bs) / bw) + e((-u - Bs) / bw) - 2 * e(-Bs / bw))


res, coef = {}, {}
for mi, mt in enumerate(mats):
    w = windows("test", mt); idx = np.random.default_rng(0).permutation(len(w["B"]))[:args.n]
    B, H, T = (torch.tensor(w[x][idx], device=DEV) for x in "BHT")
    with torch.no_grad():
        m(B, H, T, torch.full((len(B),), mi, device=DEV), args.k)
        (u, s, HE, p0, c0, al, r, ww, L, Lc, k, gam), kw = cap["scan"][0][:12], cap["scan"][1]
        mm = kw["m"]                                                            # gamma (1,N,W,K), m (1,N,W)
        h = _scan(u, s, HE, p0, c0, al, r, ww, L, Lc, k, gamma=gam, m=mm)       # model output before the lag (1,N,W)
        P = lambda n: m.bb.P(n, torch.full((len(B),), mi, device=DEV))
        t = ((T - 50) / 25)[None, :, None]; tp = torch.cat([torch.ones_like(t), t, torch.zeros_like(t)], -1)
        Z = cap["z"][None]; g = lambda o: Z[..., m.outs.index(o)] if o in m.outs else torch.zeros_like(u)
        q = m.last_q[..., :1] if m.last_q is not None else 0.0
        a1 = (P("a1") * tp).sum(-1, keepdim=True) * torch.exp(g("a1")) * (1 + q)
        a3 = 100 * ((P("a3") * tp).sum(-1, keepdim=True) + g("a3")) * (1 + q)
        Bs = ((P("bs") * tp).sum(-1, keepdim=True) + 0.05 * g("bs")).clamp(min=0.15)
        bw = torch.exp(P("log_bw")[..., :1] + g("bw"))
        # recurrence of scan_tv.scan_ref with the states: observer during the warm-up, energy terms in the predicted part
        pump = torch.zeros_like(u[..., 0]); pump_pos = torch.zeros_like(pump); diss = torch.zeros_like(pump); ein = torch.zeros_like(pump)
        pp = p0; cc = c0[..., None]; hprev = HE[..., :1]; vprev = None       # vprev: drive of the previous step
        for j in range(u.shape[-1]):
            v = u[..., j:j + 1] + gam[..., j, :] * hprev
            x = v - pp
            pn = pp + al[..., j, :] * (torch.relu(x - r[..., j, :]) - torch.relu(-x - r[..., j, :]))
            hj = s[..., j:j + 1] + mm[..., j:j + 1] * (ww[..., j, :] * pn).sum(-1, keepdim=True) + cc
            if j < k:
                e = HE[..., j:j + 1] - hj; pn = pn + L * e; cc = cc + Lc[..., None] * e; hj = HE[..., j:j + 1]
            elif j > k:
                th = lambda jj: (a1[..., jj], a3[..., jj], Bs[..., jj], bw[..., jj], mm[..., jj, None] * ww[..., jj, :])
                def E(jj, uu, vv, ppp):
                    """Stored energy at step jj: reversible curve and play inductors (drive vv, play states ppp)."""
                    A1, A3, BS, BW, mw = th(jj)
                    return W_rev(uu, A1, A3, BS, BW) + (mw * vv ** 2 / 2).sum(-1) + (-mw * (vv - ppp) ** 2 / 2).sum(-1)
                dp = E(j, u[..., j - 1], vprev, pp) - E(j - 1, u[..., j - 1], vprev, pp)
                pump += dp; pump_pos += dp.clamp(min=0)
                diss += (-(mm[..., j, None] * ww[..., j, :]) * (v - (pn + pp) / 2) * (pn - pp)).sum(-1)
                ein += hj[..., 0] * (u[..., j] - u[..., j - 1])
            vprev = v; pp = pn; hprev = hj
        rel = (pump / diss.clamp(min=1e-12))[0].cpu().numpy(); relp = (pump_pos / diss.clamp(min=1e-12))[0].cpu().numpy()
        res[mt] = {"pump_over_diss_median": float(np.median(rel)), "pump_over_diss_p95": float(np.percentile(rel, 95)),
                   "pump_pos_over_diss_median": float(np.median(relp)), "pump_pos_over_diss_p95": float(np.percentile(relp, 95)),
                   "frac_pump_gt_diss": float((rel > 1).mean()), "total_pump_over_diss": float(pump.sum() / diss.sum()),
                   "diss_over_input": float(diss.sum() / ein.sum())}
        print(mt, {kk: round(vv, 4) for kk, vv in res[mt].items()}, flush=True)
        sl = slice(0, 4)                                                        # coefficient trajectories of 4 windows
        coef[mt] = dict(B=B[sl].cpu().numpy(), T=T[sl].cpu().numpy(), u=u[0, sl].cpu().numpy(),
                        alpha=al[0, sl, :, 0].cpu().numpy(), beta=al[0, sl, :, -2:].cpu().numpy(), r=r[0, sl, :, :m.M].cpu().numpy(),
                        w=(mm[..., None] * ww)[0, sl].cpu().numpy(), a1=a1[0, sl].cpu().numpy(),
                        fid=w["fid"][idx][sl])
tot = {kk: float(np.mean([r[kk] for r in res.values()])) for kk in next(iter(res.values()))}
print("mean over materials", {kk: round(vv, 4) for kk, vv in tot.items()})
json.dump({"ckpt": args.ckpt, "k": args.k, "per_mat": res, "mean": tot}, open(ROOT / f"results/hyper_energy_{args.ckpt}.json", "w"), indent=1)
np.savez_compressed(ROOT / f"results/hyper_coef_{args.ckpt}.npz", outs=np.array(m.outs), mats=np.array(mats),
                    **{f"{mt}:{kk}": vv for mt, d in coef.items() for kk, vv in d.items()})
