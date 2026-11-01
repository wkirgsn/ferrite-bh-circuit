"""Step-size consistency of the law read as an ODE: open-loop simulation from the demagnetized state on test windows at
the native step and with 8 sub-steps (spice_check.law_sim); relative RMS difference per material (median, max).
Usage: python ode_check.py <ckpt> <variant> [n windows]"""
import sys, numpy as np, torch
from mc2lib import windows, ROOT
from compact_m import CompactM, DEV
from spice_check import coeffs, law_sim

ck = torch.load(ROOT / f"models/{sys.argv[1]}.pt", map_location=DEV); e = ck["names"].index(sys.argv[2])
m = CompactM(ck["variants"], len(ck["mats"]), ck["M"], ck["J"]).to(DEV); m.load_state_dict(ck["sd"], strict=False)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20; rng = np.random.default_rng(1); alld = []
for mid, mt in enumerate(ck["mats"]):
    w = windows("test", mt); d = []
    for q in rng.choice(len(w["B"]), n, replace=False):
        k, c = coeffs(m, e, mid, w["B"][q], w["H"][q], float(w["T"][q])); u = c["u"].astype(np.float64)
        u = np.r_[u[0] * (1 - np.cos(np.linspace(0, np.pi, 300))) / 2, u[1:]]
        H1, H8 = law_sim(k, u, 1), law_sim(k, u, 8)
        d.append(100 * np.sqrt(((H1 - H8)[299:] ** 2).mean() / (H8[299:] ** 2).mean()))
    print(f"{mt:5s} law(dt) vs law(dt/8): median {np.median(d):6.2f} %  max {np.max(d):7.2f} %  alpha {k['alpha']:.3f} beta {np.round(k['beta'], 4)}", flush=True)
    alld += d
print(f"all {len(alld)} windows: median {np.median(alld):.2f} %, 90th percentile {np.percentile(alld, 90):.2f} %, max {np.max(alld):.2f} %")
