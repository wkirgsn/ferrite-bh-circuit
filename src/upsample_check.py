"""Step size and the measurement correction: open-loop simulation from the demagnetized state on test windows, with the
correction u = B + tau1 B' + tau2 B'' either computed at 16 MHz and interpolated (as in ode_check.py) or computed on the
finer grid from the linearly interpolated B. Relative RMS difference to the native step, per upsampling factor.
Usage: python upsample_check.py <ckpt> <variant> [material=3C94] [n windows=20]"""
import sys, numpy as np, torch
from mc2lib import windows, ROOT
from compact_m import CompactM, DEV
from spice_check import coeffs, law_sim

ck = torch.load(ROOT / f"models/{sys.argv[1]}.pt", map_location=DEV); e = ck["names"].index(sys.argv[2])
m = CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]).to(DEV); m.load_state_dict(ck["sd"], strict=False)
mt = sys.argv[3] if len(sys.argv) > 3 else "3C94"; n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
mid = ck["mats"].index(mt); w = windows("test", mt); rng = np.random.default_rng(1)
ramp = lambda x: np.r_[x[0] * (1 - np.cos(np.linspace(0, np.pi, 300))) / 2, x[1:]]
rel = lambda a, b: 100 * np.sqrt(((a - b)[299:] ** 2).mean() / (b[299:] ** 2).mean())
d = {s: ([], []) for s in (2, 4, 8)}
for q in rng.choice(len(w["B"]), n, replace=False):
    k, c = coeffs(m, e, mid, w["B"][q], w["H"][q], float(w["T"][q]))
    B = w["B"][q].astype(np.float64); u = c["u"].astype(np.float64)
    d1 = np.gradient(B); d2 = np.gradient(d1)
    tau = np.linalg.lstsq(np.c_[d1, d2], u - B, rcond=None)[0]               # (tau1, tau2) in samples, from the captured u
    Br = ramp(B); H1 = law_sim(k, ramp(u), 1)
    for s in d:
        tf = np.arange((len(Br) - 1) * s + 1) / s; Bf = np.interp(tf, np.arange(len(Br)), Br)
        f1 = np.gradient(Bf) * s; f2 = np.gradient(f1) * s
        uf = Bf + tau[0] * f1 + tau[1] * f2
        d[s][0].append(rel(law_sim(k, ramp(u), s), H1)); d[s][1].append(rel(law_sim(k, uf, s, fine=True), H1))
print(f"{mt}, {n} test windows, tau = {np.round(tau, 3)} samples (last window)")
for s, (a, b) in d.items():
    print(f"{s}-fold: u at 16 MHz, interpolated: median {np.median(a):.2f} %   u on the fine grid: median {np.median(b):.2f} %")
