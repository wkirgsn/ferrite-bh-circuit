"""Check the NumPy reference implementation (ferrite_bh/model.py) against the PyTorch model on test windows of every
material (warm-up k = 100, 500, 900): relative RMS difference of the predicted part, and the test SRE of both.
Usage: python check_numpy.py [n windows per material=24]"""
import sys, numpy as np, torch
from mc2lib import ROOT, windows
from loader import load
sys.path.insert(0, str(ROOT)); import ferrite_bh as fb

n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
m, names, mats = load("nere"); e = names.index("final_e1"); rng = np.random.default_rng(0); worst = 0
for i, mt in enumerate(mats):
    w = windows("test", mt); idx = rng.choice(len(w["B"]), n, replace=False); d, s1, s2 = [], [], []
    for k in (100, 500, 900):
        B, H, T = w["B"][idx].astype(np.float64), w["H"][idx].astype(np.float64), w["T"][idx].astype(np.float64)
        with torch.no_grad():
            Ht = m(torch.tensor(B, dtype=torch.float32), torch.tensor(H, dtype=torch.float32), torch.tensor(T, dtype=torch.float32),
                   torch.full((n,), i), k)[e].numpy()
        Hn = fb.predict(mt, B, H[:, :k], T)
        sre = lambda a: np.sqrt(((a[:, k:] - H[:, k:]) ** 2).mean(1) / (H[:, k:] ** 2).mean(1)) * 100
        d.append(np.sqrt(((Hn[:, k:] - Ht[:, k:]) ** 2).mean(1) / (H[:, k:] ** 2).mean(1)) * 100); s1.append(sre(Ht)); s2.append(sre(Hn))
    d = np.concatenate(d); worst = max(worst, d.max())
    print(f"{mt:6s} NumPy vs PyTorch: median {np.median(d):.1e} %, max {d.max():.1e} %   SRE {np.mean(s1):.3f} / {np.mean(s2):.3f} %", flush=True)
print(f"max difference over all windows {worst:.1e} % of rms H")
