"""Per-material test statistics of one model variant and of the published RHINO-MAG GRU8 for the main results table:
mean and 95th percentile of SRE and NERE (mc2lib.metrics), each per warm-up scenario and then averaged over the three
scenarios, as in the challenge's evaluation; plus the mean SRE per scenario. Runs on the CPU with the reference scan when
the GPU is busy. Usage: python main_stats.py <ckpt> <variant>   -> results/stats_<ckpt>_<variant>.json"""
import sys, json, numpy as np
import compact_m, scan
from compact_m import evaluate_M, DEV
from mc2lib import ROOT, windows
from official_eval import prctile

if DEV.type == "cpu":
    compact_m.scan = scan.scan_ref
ckn, var = sys.argv[1:3]
from loader import load  # noqa: E402  (after the CPU fallback)
m, names, mats = load(ckn); ck = {"names": names, "mats": mats}       # classical or scheduled circuit
e = ck["names"].index(var)
stat = lambda S, N: dict(sre=100 * float(np.mean([s.mean() for s in S])), sre_p95=float(np.mean([prctile(100 * s, 95) for s in S])),
                         nere=100 * float(np.mean([n.mean() for n in N])), nere_p95=float(np.mean([prctile(100 * n, 95) for n in N])),
                         scen=[100 * float(s.mean()) for s in S])
out = {}
for i, mt in enumerate(ck["mats"]):
    res = evaluate_M(m, windows("test", mt), i)
    rr = np.load(ROOT / ("data/rhino_res.npy" if mt == "3C94" else f"data/{mt}_rhino_res.npy"), allow_pickle=True).item()
    out[mt] = {"model": stat([res[s][0][e] for s in res], [res[s][1][e] for s in res]),
               "gru": stat([rr[s]["sre"] for s in rr], [rr[s]["nere"] for s in rr])}
    print(mt, {k: {q: round(v, 2) for q, v in d.items() if isinstance(v, float)} for k, d in out[mt].items()}, flush=True)
json.dump(out, open(ROOT / f"results/stats_{ckn}_{var}.json", "w"), indent=1)
