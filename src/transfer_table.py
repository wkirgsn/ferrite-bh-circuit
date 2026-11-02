"""Table of the MC2 -> MC1 specimen-transfer experiment from results/<prefix>_<mat>_mc1.txt (validation lines of the
finetune_m.py output: val SRE after refit).
Usage: python transfer_table.py [prefix=ftf]   (ftf: final 32-parameter model; ft70: earlier 61-parameter version)   -> paper/tables/transfer.typ"""
import sys, re, glob, numpy as np
from pathlib import Path
from mc2lib import ROOT
pre = sys.argv[1] if len(sys.argv) > 1 else "ftf"
G = ["none", "tau", "amp", "tau_amp", "all"]
rows = {}
for f in sorted(glob.glob(str(ROOT / f"results/{pre}_*_mc1.txt"))):
    mat = Path(f).name.removeprefix(f"{pre}_").removesuffix("_mc1.txt"); r = {}
    for ln in open(f):
        m = re.match(r"val \S+ \(after\) (\S+)\s+n=\s*\d+\s+SRE avg\s+([\d.]+)", ln)
        if m: r[m.group(1)] = float(m.group(2))
    if len(r) == len(G): rows[mat] = r
import pandas as pd  # noqa: E402
def turns(m):
    """Number of turns of the MC2 and of the MC1 specimen of material m."""
    x = pd.read_excel(ROOT / f"data/info/{m}_Test_Info.xlsx", header=None).set_index(0)[1]
    return int(x["N1"]), int(np.load(ROOT / f"data/mc1/{m}.npz")["N1"])
lines = [f"  [{m}], [{turns(m)[0]} / {turns(m)[1]}], " + ", ".join(f"[{rows[m][g]:.1f}]" for g in G) + "," for m in rows]
mean = [np.mean([rows[m][g] for m in rows]) for g in G] if rows else [float("nan")] * len(G)
if not rows: lines = ["  // PENDING: transfer runs (queue22)"]
tab = ("#figure(\n  table(\n    columns: 7, align: (left,) + (right,) * 6, stroke: none,\n    table.hline(),\n"
       "    table.header([*Material*], [$N$ MC2/MC1], [none], [meas. (5)], [ampl. (14)], [meas. + ampl. (19)], [all (32)]),\n    table.hline(),\n"
       + "\n".join(lines) + "\n    table.hline(),\n    [*mean*], [], " + ", ".join(f"[*{x:.1f}*]" for x in mean) + ",\n    table.hline(),\n  ),\n"
       "  caption: [Transfer of the model from the MC2 specimen to the MC1 specimen of each material (same core shape; $N$: turns):\n"
       "  validation SRE (%) on the MC1 recording (25–70 °C) after refitting only the named parameter groups (number of\n"
       "  parameters in brackets); all other parameters keep their MC2 values.],\n) <tab:transfer_all>\n")
open(ROOT / "paper/tables/transfer.typ", "w").write(tab); print(tab)
