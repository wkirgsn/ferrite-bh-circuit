"""Evaluation-only check of a model-based warm-up estimator: an extended Kalman filter (EKF) on the trained model in
place of the learned estimator (prior coefficients g, g2, mismatch gains K0, offset prior cpr, observer gains L and the
Bayesian-ridge amplitude factors q, qh with prior lq, lqk). The model and its parameters are unchanged.

EKF state x = (p_1..p_M, z_1..z_J, c, q, qh). During the warm-up the feedback uses the measured (de-lagged) H, as in
the model's own observer, so the transition is diagonal: dp_i/dp_i = 1 inside the dead zone, 1 - alpha outside;
1 - beta_j for the relaxations; c, q, qh constant (c a random walk). Output
    H = s + q (s - h_sat) + (1 + qh) m sum_i w_i p_i + c
(s the reversible curve, h_sat its knee, as the model's amplitude estimate). Prior without learned parameters: plays at
the centre of their band (variance r_i^2/3), relaxations at their ramp steady state u0 - u0' (1 - beta)/beta, offset
diffuse, q, qh ~ N(0, sq^2). Shared hyperparameters (all materials): rho (measurement noise / rms H of the warm-up),
kap (play process noise / r_i per step), rc (offset random walk / rms H per step), sq. After the warm-up the model
runs open loop from the filtered state, exactly as with the learned estimator, and the H lag is applied.
Usage: python ekf_eval.py <ckpt> <variant> [--tune]   -> results/ekf_<ckpt>_<variant>.json"""
import argparse, itertools, json, time
import numpy as np, torch
from mc2lib import *
import compact_m, scan
from compact_m import CompactM
compact_m.scan = scan.scan_ref
DEV = torch.device("cpu")


def load_single(ckn, var):
    ck = torch.load(ROOT / f"models/{ckn}.pt", map_location=DEV); e = ck["names"].index(var)
    m = CompactM([ck["variants"][e]], len(ck["mats"]), ck["M"], ck["J"])
    m.load_state_dict({k: v[e:e + 1] for k, v in ck["sd"].items() if k.startswith("p.")}, strict=False)
    return m.eval(), ck["mats"]


def terms(m, w, mid, k):
    """Model terms of the windows w (one material) at warm-up k, the model's own prediction and its learned estimator."""
    B, H, T = (torch.tensor(w[n], dtype=torch.float32) for n in ("B", "H", "T"))
    dt = torch.tensor(w["dt"]) if "dt" in w else None
    mi = torch.full((len(B),), mid)
    m.capture = True
    with torch.no_grad():
        out = m(B, H, T, mi, k, dt=dt)[0]
    c = {n: v[0].double() for n, v in m.cap.items()}; m.capture = False
    c["q2"] = m.last_q[0].double(); c["out"] = out.double(); c["H"] = H.double(); c["B"] = B.double()
    hla = c["hla"]; c["HE"] = (c["H"] - hla * torch.cat([c["H"][:, :1], c["H"][:, :-1]], -1)) / (1 - hla)
    P = lambda n: m.P(n, mi)[0].detach().double()
    for n in ("g", "g2", "K0", "cpr", "L"):
        c[n] = P(n)
    return c


def run(c, k, p, cc, gains=None, ekf=None, q=None):
    """Warm-up (observer: constant gains or EKF) and open-loop prediction. p (N,K) states before step 0, cc (N,1)."""
    M = c["r"].shape[-1]; J = c["beta"].shape[-1]
    r = torch.cat([c["r"], torch.zeros_like(c["beta"])], -1); al = torch.cat([c["alpha"].expand(-1, M), c["beta"]], -1)
    wv = torch.cat([c["w"], c["cz"]], -1); gam = c["gamma"]; u, HE = c["u"], c["HE"]
    s0, hs, mm = c["static"], c["hsat"], c["mmod"]
    N, W = u.shape
    if ekf is not None:                                   # augmented state (p, c, q, qh), covariance Pc (N,K+3,K+3)
        x = torch.cat([p, cc, ekf.get("q0", torch.zeros(N, 2, dtype=p.dtype))], -1); Pc = ekf["P0"].clone()
        Kp = p.shape[-1]; I = torch.eye(Kp + 3, dtype=p.dtype)
    else:
        qq, qh = q[:, :1], q[:, 1:]
    outs = []; hp = HE[:, :1]
    for j in range(W):
        v = u[:, j:j + 1] + gam * hp
        if ekf is not None and j < k:
            p, cc = x[:, :Kp], x[:, Kp:Kp + 1]
        dz = v - p
        drag = (dz.abs() > r).to(p.dtype)
        p = p + al * (torch.relu(dz - r) - torch.relu(-dz - r))
        if ekf is not None and j < k:
            x = torch.cat([p, x[:, Kp:]], -1)
            F = torch.cat([1 - al * drag, torch.ones(N, 3, dtype=p.dtype)], -1)
            Pc = F[:, :, None] * Pc * F[:, None, :] + ekf["Q"]
            qq, qh = x[:, Kp + 1:Kp + 2], x[:, Kp + 2:Kp + 3]
            osum = (wv * p).sum(-1, keepdim=True)
            h = s0[:, j:j + 1] + qq * (s0[:, j:j + 1] - hs[:, j:j + 1]) + (1 + qh) * mm[:, j:j + 1] * osum + x[:, Kp:Kp + 1]
            C = torch.cat([(1 + qh) * mm[:, j:j + 1] * wv, torch.ones(N, 1, dtype=p.dtype), s0[:, j:j + 1] - hs[:, j:j + 1],
                           mm[:, j:j + 1] * osum], -1)                       # (N,K+3)
            PC = (Pc @ C[:, :, None])[..., 0]; S = (PC * C).sum(-1, keepdim=True) + ekf["R"]
            G = PC / S; e = HE[:, j:j + 1] - h
            x = x + G * e
            A = I - G[:, :, None] * C[:, None, :]
            Pc = A @ Pc @ A.transpose(1, 2) + G[:, :, None] * G[:, None, :] * ekf["R"][:, :, None]   # Joseph form
            h = HE[:, j:j + 1]
            if j == k - 1:
                p, cc = x[:, :Kp], x[:, Kp:Kp + 1]
                qq, qh = x[:, Kp + 1:Kp + 2].clamp(-0.5, 0.5), x[:, Kp + 2:Kp + 3].clamp(-0.5, 0.5)
        else:
            h = s0[:, j:j + 1] + qq * (s0[:, j:j + 1] - hs[:, j:j + 1]) + (1 + qh) * mm[:, j:j + 1] * (wv * p).sum(-1, keepdim=True) + cc
            if j < k:                                     # learned constant gains
                e = HE[:, j:j + 1] - h
                p = p + gains[0] * e; cc = cc + gains[1] * e; h = HE[:, j:j + 1]
        outs.append(h); hp = h
    out = torch.cat(outs, -1)
    return CompactM.lag(out, c["hla"], c["H"][:, :1])


def learned(c, k):
    """The model's learned estimator, re-implemented (check against the model's own output)."""
    M = c["r"].shape[-1]; J = c["beta"].shape[-1]; u = c["u"]; u0 = u[:, :1]; r = c["r"]
    p = u0 - r * torch.tanh(c["g2"] * u0 / r + torch.sign(u[:, 5:6] - u0) * c["g"]); z = u0.expand(-1, J)
    e0 = c["HE"][:, :1] - c["static"][:, :1] - (c["w"] * p).sum(-1, keepdim=True) - (c["cz"] * z).sum(-1, keepdim=True)
    K0 = c["K0"]; p = p + 1e-3 * K0[:, :M] * e0; z = z + 1e-3 * K0[:, M:M + J] * e0; cc = c["cpr"] + K0[:, -1:] * e0
    return run(c, k, torch.cat([p, z], -1), cc, gains=(1e-3 * c["L"][:, :M + J], c["L"][:, -1:]), q=c["q2"])


def ekf(c, k, hp, prior="model", amp="ekf"):
    """EKF estimator with the shared hyperparameters hp = (rho, kap, rc, sq). Diagnostics: prior="learned" starts from
    the learned prior (g, g2, K0, cpr) with the same covariance; amp="learned" fixes q, qh at the learned ridge estimate."""
    rho, kap, rc, sq = hp
    M = c["r"].shape[-1]; J = c["beta"].shape[-1]; u = c["u"]; u0 = u[:, :1]; N = u.shape[0]
    du0 = (u[:, 5:6] - u0) / 5; b = c["beta"]
    zs = du0 * (1 - b) / b                                # ramp lag of the relaxations
    p = torch.cat([u0.expand(-1, M), u0 - zs], -1); cc = torch.zeros(N, 1, dtype=u.dtype)
    if prior == "learned":
        r = c["r"]; p = u0 - r * torch.tanh(c["g2"] * u0 / r + torch.sign(u[:, 5:6] - u0) * c["g"]); z = u0.expand(-1, J)
        e0 = c["HE"][:, :1] - c["static"][:, :1] - (c["w"] * p).sum(-1, keepdim=True) - (c["cz"] * z).sum(-1, keepdim=True)
        K0 = c["K0"]; p = torch.cat([p + 1e-3 * K0[:, :M] * e0, z + 1e-3 * K0[:, M:M + J] * e0], -1); cc = c["cpr"] + K0[:, -1:] * e0
    sH = c["HE"][:, :k].pow(2).mean(-1, keepdim=True).sqrt()
    vq = 1e-12 if amp == "learned" else sq ** 2
    var0 = torch.cat([c["r"] ** 2 / 3, zs ** 2 + 1e-8, (10 * sH) ** 2, torch.full((N, 2), vq, dtype=u.dtype)], -1)
    Q = torch.cat([(kap * c["r"]) ** 2, torch.full((N, J), 1e-12, dtype=u.dtype), (rc * sH) ** 2, torch.zeros(N, 2, dtype=u.dtype)], -1)
    e = dict(P0=torch.diag_embed(var0), Q=torch.diag_embed(Q), R=(rho * sH) ** 2)
    if amp == "learned":
        e["q0"] = c["q2"]
    return run(c, k, p, cc, ekf=e)


def score(Hp, c, k):
    sre, nere = metrics(Hp.numpy(), c["H"].numpy(), c["B"].numpy(), k)
    return sre, nere


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("variant")
    ap.add_argument("--tune", action="store_true", help="grid search of the shared hyperparameters on validation windows")
    ap.add_argument("--nval", type=int, default=120, help="validation windows per material for tuning")
    ap.add_argument("--hp", default="0.03,0.01,0.001,0.2", help="rho,kap,rc,sq when not tuning")
    args = ap.parse_args()
    torch.set_num_threads(8)
    m, mats = load_single(args.ckpt, args.variant)
    t0 = time.time()
    # check: the re-implemented learned estimator reproduces the model
    c = terms(m, {n: v[:20] for n, v in windows("val", mats[0]).items()}, 0, 500)
    print("re-implementation vs model: max |dH| %.2e of %.0f A/m" % ((learned(c, 500) - c["out"]).abs().max(), c["H"].abs().max()), flush=True)
    hp = tuple(float(x) for x in args.hp.split(","))
    res = {"hp": hp}
    if args.tune:
        val = {}
        for i, mt in enumerate(mats):
            w = windows("val", mt); sel = np.random.default_rng(0).permutation(len(w["B"]))[:args.nval]
            val[mt] = {kk: terms(m, {n: v[sel] for n, v in w.items()}, i, kk) for kk in SCENARIOS.values()}
        base = np.mean([[score(learned(val[mt][kk], kk), val[mt][kk], kk)[0].mean() for kk in SCENARIOS.values()] for mt in mats])
        print(f"val learned estimator: mean SRE {100 * base:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        grid = list(itertools.product([0.01, 0.03, 0.1], [0.003, 0.01, 0.03], [0.0, 1e-3, 1e-2], [0.1, 0.3]))
        best = (9, None)
        for g in grid:
            sc = np.mean([[score(ekf(val[mt][kk], kk, g), val[mt][kk], kk)[0].mean() for kk in SCENARIOS.values()] for mt in mats])
            print(f"val EKF rho {g[0]} kap {g[1]} rc {g[2]} sq {g[3]}: mean SRE {100 * sc:.2f}  ({time.time() - t0:.0f}s)", flush=True)
            if sc < best[0]: best = (sc, g)
        hp = best[1]; res.update(hp=hp, val_learned=100 * base, val_ekf=100 * best[0])
        print("best", hp, flush=True)
    for i, mt in enumerate(mats):
        wt = windows("test", mt); res[mt] = {}
        for sc_name, kk in SCENARIOS.items():
            c = terms(m, select_k(wt, kk), i, kk)
            a, b = score(learned(c, kk), c, kk), score(ekf(c, kk, hp), c, kk)
            res[mt][sc_name] = {"learned": [100 * a[0].mean(), 100 * a[1].mean()], "ekf": [100 * b[0].mean(), 100 * b[1].mean()]}
        L = np.mean([v["learned"][0] for v in res[mt].values()]); Ek = np.mean([v["ekf"][0] for v in res[mt].values()])
        print(f"test {mt}: SRE learned {L:.2f}  EKF {Ek:.2f}  |  " + "  ".join(f"{s}: {v['learned'][0]:.2f}/{v['ekf'][0]:.2f}" for s, v in res[mt].items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    for key in ("learned", "ekf"):
        res[f"mean_{key}"] = float(np.mean([[v[key][0] for v in res[mt].values()] for mt in mats]))
    print(f"test mean SRE: learned {res['mean_learned']:.2f}  EKF {res['mean_ekf']:.2f}")
    json.dump(res, open(ROOT / f"results/ekf_{args.ckpt}_{args.variant}.json", "w"), indent=1)
