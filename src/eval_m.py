"""Evaluate a CompactM checkpoint on the test windows of each of its materials; per-file SRE vs published RHINO GRU8.
Usage: python eval_m.py <checkpoint> [variant ...]   -> results/<checkpoint>.json, a line per variant in results/runs.jsonl"""
import sys, json, numpy as np, torch
from mc2lib import *
from compact_m import CompactM, evaluate_M, DEV
from common import summarize

from loader import load
m, names, mats = load(sys.argv[1]); ck = {"names": names, "mats": mats}   # CompactM (older checkpoints lack later options) or hybrid
sel = [ck["names"].index(n) for n in sys.argv[2:]] if len(sys.argv) > 2 else range(m.E)
out = {}  # results/<ckpt>.json: {mat: {variant: {sre, scen, files, nparams}, "GRU8": {...}}}
for i, mt in enumerate(ck["mats"]):
    w = windows("test", mt); res = evaluate_M(m, w, i)
    summarize(res, ck["names"], m.n_params(), f"TEST {mt}")
    rr = np.load(ROOT / ("data/rhino_res.npy" if mt == "3C94" else f"data/{mt}_rhino_res.npy"), allow_pickle=True).item()
    print(f"TEST {mt} RHINO GRU8 (published)   n=336  SRE avg {100*np.mean([rr[s]['sre'].mean() for s in rr]):5.2f} | " + " ".join(f"{100*rr[s]['sre'].mean():5.2f}" for s in rr))
    fids = np.unique(w["fid"]) if "k" not in w else []   # official test sets have no file information
    print(f"  per-file SRE avg over scenarios:   " + "  ".join(f"f{f}" for f in fids))
    print(f"  GRU8 published               " + " ".join(f"{100*np.mean([rr[s]['sre'][w['fid']==f].mean() for s in rr]):4.1f}" for f in fids))
    for e in sel:
        print(f"  {ck['names'][e]:28s} " + " ".join(f"{100*np.mean([res[s][0][e][w['fid']==f].mean() for s in res]):4.1f}" for f in fids))
    rec = lambda S, N, np_: dict(sre=100 * float(np.mean([x.mean() for x in S])), scen=[100 * float(x.mean()) for x in S],
                                 nere=[100 * float(x.mean()) for x in N], nparams=np_,
                                 files={int(f): 100 * float(np.mean([x[w["fid"] == f].mean() for x in S])) for f in fids})
    if "k" in w:
        print(f"  {mt}: official MC2 test set ({len(w['k'])} windows, one warm-up length each)")
    out[mt] = {ck["names"][e]: rec([res[s][0][e] for s in res], [res[s][1][e] for s in res], m.n_params()[e] // m.nm) for e in sel}
    rv = evaluate_M(m, windows("val", mt), i)                     # validation SRE, used to select the variant per material
    for e in sel:
        out[mt][ck["names"][e]]["val"] = 100 * float(np.mean([rv[s][0][e].mean() for s in rv]))
    out[mt]["GRU8"] = rec([rr[s]["sre"] for s in rr], [rr[s].get("nere", rr[s]["sre"] * np.nan) for s in rr], 336)
(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / f"results/{sys.argv[1]}.json", "w"), indent=1)
import runlog                                                             # one record per variant in results/runs.jsonl
cfg = runlog.config(torch.load(ROOT / f"models/{sys.argv[1]}.pt", map_location="cpu", weights_only=False))
for e in sel:
    runlog.append(runlog.test_record(sys.argv[1], ck["names"][e], out, cfg))
