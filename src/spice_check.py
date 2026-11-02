"""Equivalent circuit of the law, simulated with ngspice, against the law itself (both open-loop from the demagnetized
state, no warm-up estimator, no measurement correction: the circuit is driven with the corrected flux u).

Unit core (N = A_e = l_e = 1): flux linkage = u, terminal current = H. Elements (k: operator, rates per sample -> 1/s with
kappa = -ln(1 - rate)/dt):
  reversible curve     nonlinear inductor  i = a1 u + a3 u^3 + 10 [e^((u-Bs)/b) - e^((-u-Bs)/b)]
  operator k           driven through a current-controlled voltage source gamma_k di/dt (feedback v_k = u + gamma_k H);
                       behind it the linear part w_k v_k (inductor 1/w_k) in parallel with the operator branch
  play i               inductor L_i = 1/|w_i| in series with a viscous Coulomb slider: diode bridge carrying the DC
                       current I_c = |w_i| r_i, in parallel with R_i = kappa_alpha / |w_i|   (Jenkins-Bingham element)
  relaxation j         inductor L_j = 1/|c_j| in series with R_j = kappa_beta_j / |c_j|, current counted with -sign(c_j)
The law gives each operator the output w_k p_k = w_k (v_k - s_k), s_k the flux across the operator's inductor.
Usage: python spice_check.py <ckpt> <variant> <mats> [n windows per material]"""
import sys, subprocess, tempfile, os, numpy as np, torch
from mc2lib import windows, ROOT
from compact_m import CompactM, DEV
from scan import scan

NGSPICE = os.environ.get("NGSPICE") or next((str(p) for p in [ROOT / "tools/ngspice/usr/bin/ngspice"] if p.exists()), "ngspice")
DT = 62.5e-9
BRIDGE = False                     # True: sliders as diode bridges (stiff for the solver at the PWL breakpoints)


def coeffs(m, e, mid, B, H, T):
    """Law coefficients of variant e, material mid at temperature T (captured from a forward pass)."""
    m.capture = True
    with torch.no_grad():
        m(torch.tensor(B[None], device=DEV), torch.tensor(H[None], device=DEV), torch.tensor([T], device=DEV),
          torch.tensor([mid], device=DEV), 100)
    c = {n: v[e, 0].detach().cpu().numpy().astype(np.float64) for n, v in m.cap.items()}; m.capture = False
    M = len(c["r"])
    return dict(a1=float(c["a1"].ravel()[0]), a3=float(c["a3"].ravel()[0]), Bs=float(c["Bs"].ravel()[0]), bw=float(c["bw"].ravel()[0]),
                r=c["r"], alpha=float(c["alpha"].ravel()[0]), beta=c["beta"], w=c["w"], cz=c["cz"], gamma=c["gamma"], M=M), c


def hs(k, u):
    """Reversible curve h_s(u) in A/m for the circuit coefficients k."""
    return k["a1"] * u + k["a3"] * u ** 3 + 10 * (np.exp(np.minimum((u - k["Bs"]) / k["bw"], 15)) - np.exp(np.minimum((-u - k["Bs"]) / k["bw"], 15)))


def law_sim(k, u, sub=1, fine=False):
    """The law (scan kernel, observer off) from the demagnetized state on u, optionally with sub steps per sample (u
    linearly interpolated, rates 1 - (1 - a)^(1/sub)); returns H at the samples. fine: u is already given on the fine grid."""
    if fine:
        uf = np.asarray(u)
    else:
        n = len(u); t = np.arange(n); tf = np.arange((n - 1) * sub + 1) / sub; uf = np.interp(tf, t, u)
    M = k["M"]; J = len(k["beta"]); dt = 1.0 / sub
    rate = lambda a: 1 - (1 - np.clip(a, 0, 1 - 1e-9)) ** dt
    T_ = lambda x: torch.tensor(np.asarray(x, np.float32), device=DEV)
    al = T_(np.r_[np.full(M, rate(k["alpha"])), rate(k["beta"])])[None]
    rr = T_(np.r_[k["r"], np.zeros(J)])[None]; ww = T_(np.r_[k["w"], k["cz"]])[None]
    g = T_(k["gamma"])[None]; z = torch.zeros(1, device=DEV)
    uu = T_(uf)[None]; st = T_(hs(k, uf))[None]; H0 = torch.zeros_like(uu)
    with torch.no_grad():
        h = scan(uu, st, H0, torch.zeros(1, M + J, device=DEV), z, al, rr, ww, torch.zeros_like(ww), z, 1, g)
    return h[0].cpu().numpy()[::sub]


def netlist(k, u, path):
    """Write the ngspice netlist <path>.cir of the circuit with coefficients k, driven by the flux u."""
    M = k["M"]; J = len(k["beta"]); kap = lambda a: -np.log(1 - min(a, 1 - 1e-6)) / DT
    n = len(u); du = np.diff(u) / DT
    pts = ["0 0"]
    for i, v in enumerate(du):                                   # piecewise-constant voltage = exact piecewise-linear flux
        pts += [f"{i * DT + 1e-11:.12e} {v:.9e}", f"{(i + 1) * DT:.12e} {v:.9e}"]
    L = ["* equivalent circuit of the compact law (unit core)", ".model DI D(Is=1e-14 N=1)",
         "Vsrc t 0 PWL(" + " ".join(pts) + ")", "Vsense t c 0",
         "Gint 0 fl c 0 1", "Cint fl 0 1", "Rint fl 0 1e15",
         "Fx 0 x Vsense 1", "Lx x 0 1",                              # V(x) = dH/dt (mirror of H into 1 H)                            # V(fl) = flux u
         f"Bmain c 0 I = {k['a1']:.9e}*V(fl) + {k['a3']:.9e}*V(fl)*V(fl)*V(fl) + 10*(exp(min((V(fl)-{k['Bs']:.9e})/{k['bw']:.9e},15))"
         f" - exp(min((-V(fl)-{k['Bs']:.9e})/{k['bw']:.9e},15)))"]
    for j in range(M + J):
        play = j < M
        wk = k["w"][j] if play else k["cz"][j - M]
        g = k["gamma"][j]
        L += [f"Eg{j} c d{j} x 0 {-g:.9e}",            # V(d) = V(c) + gamma dH/dt
              f"Gf{j} 0 f{j} d{j} 0 1", f"Cf{j} f{j} 0 1", f"Rf{j} f{j} 0 1e15",              # flux of the drive node
              f"Blin{j} d{j} 0 I = {wk:.9e}*V(f{j})",                   # linear part w_k v_k
              f"Eb{j} e{j} 0 d{j} 0 1"]                                 # copy of the drive for the operator branch
        Lk = 1 / abs(wk)
        if play:
            Ic = abs(wk) * k["r"][j]; R = kap(k["alpha"]) * Lk
            if BRIDGE:                                                  # diode bridge + DC current source, parallel R
                L += [f"Lp{j} e{j} m{j} {Lk:.9e}", f"Vs{j} m{j} a{j} 0",
                      f"D1_{j} a{j} p{j} DI", f"D2_{j} 0 p{j} DI", f"D3_{j} n{j} a{j} DI", f"D4_{j} n{j} 0 DI",
                      f"Ic{j} p{j} n{j} {Ic:.9e}", f"Rp{j} a{j} 0 {R:.9e}"]
            else:                                                       # the same slider as a dead-zone resistor
                L += [f"Lp{j} e{j} m{j} {Lk:.9e}", f"Vs{j} m{j} a{j} 0",
                      f"Bsl{j} a{j} 0 V = {R:.9e}*(max(I(Vs{j})-{Ic:.9e},0)+min(I(Vs{j})+{Ic:.9e},0))"]
            sgn = 1.0                                                   # w < 0: -w s = |w| s = i
        else:
            R = kap(k["beta"][j - M]) * Lk
            L += [f"Lp{j} e{j} m{j} {Lk:.9e}", f"Vs{j} m{j} a{j} 0", f"Rp{j} a{j} 0 {R:.9e}"]
            sgn = -np.sign(wk)
        if play and wk > 0: sgn = -1.0
        L += [f"Fm{j} d{j} 0 Vs{j} {sgn:.1f}"]                          # branch current counted at the drive node
    L += [".options reltol=1e-5 abstol=1e-9 method=gear", f".tran {DT / 4:.6e} {(n - 1) * DT:.6e} 0 {DT / 4:.6e}",
          ".control", "run", "set wr_singlescale", f"wrdata {path}.out i(Vsense)", "quit", ".endc", ".end"]
    open(path + ".cir", "w").write("\n".join(L) + "\n")


def spice(k, u):
    """Run ngspice on the netlist of k driven by u; returns the current into the core (= H) at the samples."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "law"); netlist(k, u, p)
        r = subprocess.run([NGSPICE, "-b", p + ".cir"], capture_output=True, text=True)
        if not os.path.exists(p + ".out"): raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
        t, i = np.loadtxt(p + ".out", unpack=True)
    return np.interp(np.arange(len(u)) * DT, t, i)                       # current into the core


if __name__ == "__main__":
    ck = torch.load(ROOT / f"models/{sys.argv[1]}.pt", map_location=DEV); e = ck["names"].index(sys.argv[2])
    m = CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]).to(DEV); m.load_state_dict(ck["sd"], strict=False)
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 3; rng = np.random.default_rng(0)
    for mt in sys.argv[3].split(","):
        mid = ck["mats"].index(mt); w = windows("test", mt); res = []
        for q in rng.choice(len(w["B"]), nw, replace=False):
            k, c = coeffs(m, e, mid, w["B"][q], w["H"][q], float(w["T"][q]))
            assert np.allclose(c["static"], hs(k, c["u"]), rtol=1e-3, atol=1e-2), "static curve mismatch"
            u = c["u"].astype(np.float64)
            u = np.r_[u[0] * (1 - np.cos(np.linspace(0, np.pi, 300))) / 2, u[1:]]      # ramp up from the demagnetized state
            s0 = 299
            Hs, H1, H8 = spice(k, u), law_sim(k, u, 1), law_sim(k, u, 8)
            rel = lambda a, b: np.sqrt(((a[s0:] - b[s0:]) ** 2).mean() / (b[s0:] ** 2).mean()) * 100
            res.append((rel(Hs, H8), rel(H1, H8), rel(Hs, H1)))
            print(f"{mt:5s} win {q:5d} T {w['T'][q]:.0f}  SPICE vs law(dt/8) {res[-1][0]:5.2f} %   law(dt) vs law(dt/8) {res[-1][1]:5.2f} %"
                  f"   SPICE vs law(dt) {res[-1][2]:5.2f} %", flush=True)
        np.save(ROOT / f"results/spice_{mt}.npy", np.array(res))
